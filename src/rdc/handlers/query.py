"""Query handlers: shader_map, pipeline, bindings, shader, shaders, resources,
resource, passes, pass, events, draws, event, draw, search, info, stats, log.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from rdc.handlers._helpers import (
    _LOG_SEVERITY_MAP,
    _SECTION_MAP,
    _SHADER_STAGES,
    _VALID_LOG_LEVELS,
    STAGE_MAP,
    PipeError,
    _action_type_str,
    _build_shader_cache,
    _enum_name,
    _error_response,
    _result_response,
    _seek_replay,
    require_pipe,
)
from rdc.handlers._types import Handler

if TYPE_CHECKING:
    from rdc.daemon_server import DaemonState


def _get_flat_actions(state: DaemonState) -> list[Any]:
    """Late-bound wrapper so tests can patch rdc.daemon_server._get_flat_actions."""
    import rdc.daemon_server as ds

    return ds._get_flat_actions(state)


def _handle_shader_map(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import collect_shader_map

    _build_shader_cache(state)
    actions = state.adapter.get_root_actions()
    rows = collect_shader_map(actions, state._pipe_states_cache)
    return _result_response(request_id, {"rows": rows}), True


def _handle_pipeline(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    from rdc.services.query_service import pipeline_row

    section = params.get("section")
    if section is not None:
        section = str(section).lower()
        if section not in _SHADER_STAGES and section not in _SECTION_MAP:
            return _error_response(request_id, -32602, "invalid section"), True
    try:
        eid, pipe_state = require_pipe(params, state, request_id)
    except PipeError as exc:
        return exc.response, True
    if section is not None and section in _SECTION_MAP:
        from rdc.handlers.pipe_state import HANDLERS as PIPE_HANDLERS

        handler = PIPE_HANDLERS.get(_SECTION_MAP[section])
        if handler is not None:
            pipe_result: tuple[dict[str, Any], bool] = handler(
                request_id, {"_token": params.get("_token", ""), "eid": eid}, state
            )
            return pipe_result
        return _error_response(request_id, -32602, "invalid section"), True
    row = pipeline_row(state.current_eid, state.api_name, pipe_state, section=section)
    return _result_response(request_id, {"row": row}), True


def _handle_bindings(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    from rdc.services.query_service import bindings_rows

    try:
        _eid, pipe_state = require_pipe(params, state, request_id)
    except PipeError as exc:
        return exc.response, True
    rows = bindings_rows(state.current_eid, pipe_state)

    set_filter = params.get("set")
    if set_filter is not None:
        rows = [r for r in rows if r.get("set") == int(set_filter)]
    binding_index = params.get("binding")
    if binding_index is not None:
        rows = [r for r in rows if r.get("slot") == int(binding_index)]

    return _result_response(request_id, {"rows": rows}), True


def _handle_shader(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    from rdc.services.query_service import shader_row

    stage = str(params.get("stage", "ps")).lower()
    if stage not in STAGE_MAP:
        return _error_response(request_id, -32602, "invalid stage"), True
    try:
        _eid, pipe_state = require_pipe(params, state, request_id)
    except PipeError as exc:
        return exc.response, True
    row = shader_row(state.current_eid, pipe_state, stage)
    if params.get("reflect"):
        stage_val = STAGE_MAP[stage]
        refl = pipe_state.GetShaderReflection(stage_val)
        if refl is not None:
            row["reflection"] = {
                "inputs": [
                    {
                        "name": getattr(sig, "varName", ""),
                        "type": _enum_name(getattr(sig, "compType", "")),
                        "location": getattr(sig, "regIndex", 0),
                    }
                    for sig in getattr(refl, "inputSignature", [])
                ],
                "outputs": [
                    {
                        "name": getattr(sig, "varName", ""),
                        "type": _enum_name(getattr(sig, "compType", "")),
                        "location": getattr(sig, "regIndex", 0),
                    }
                    for sig in getattr(refl, "outputSignature", [])
                ],
                "cbuffers": [
                    {
                        "name": cb.name,
                        "slot": getattr(cb, "fixedBindNumber", getattr(cb, "bindPoint", 0)),
                        "vars": len(getattr(cb, "variables", [])),
                    }
                    for cb in getattr(refl, "constantBlocks", [])
                ],
            }
    return _result_response(request_id, {"row": row}), True


def _handle_shaders(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    _build_shader_cache(state)
    rows = [
        {"shader": sid, "stages": ",".join(sorted(meta["stages"])), "uses": meta["uses"]}
        for sid, meta in sorted(state.shader_meta.items())
    ]
    stage_filter = params.get("stage")
    if stage_filter:
        stage_filter = stage_filter.lower()
        rows = [r for r in rows if stage_filter in r["stages"].lower().split(",")]
    return _result_response(request_id, {"rows": rows}), True


def enrich_resource_row(row: dict[str, Any], state: DaemonState) -> dict[str, Any]:
    """Merge texture/buffer dimensions and size into a base resource row."""
    rid = int(row.get("id", 0))
    tex = state.tex_map.get(rid)
    if tex is not None:
        fmt = getattr(tex, "format", None)
        row["width"] = tex.width
        row["height"] = tex.height
        row["format"] = fmt.Name() if fmt is not None and hasattr(fmt, "Name") else ""
        row["size"] = getattr(tex, "byteSize", 0)
        return row
    buf = state.buf_map.get(rid)
    if buf is not None:
        row["size"] = getattr(buf, "length", getattr(buf, "byteSize", 0))
    return row


def _handle_resources(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import get_resources

    rows = get_resources(state.adapter)
    type_filter = params.get("type")
    name_filter = params.get("name")
    sort_by = params.get("sort", "id")
    if type_filter:
        rows = [r for r in rows if r["type"].lower() == type_filter.lower()]
    if name_filter:
        name_lower = name_filter.lower()
        rows = [r for r in rows if name_lower in r["name"].lower()]
    if sort_by in ("name", "type"):
        rows.sort(key=lambda r: r[sort_by].lower())
    rows = [enrich_resource_row(r, state) for r in rows]
    return _result_response(request_id, {"rows": rows}), True


def _handle_resource(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import get_resource_detail

    resid = int(params.get("id", 0))
    detail = get_resource_detail(state.adapter, resid)
    if detail is None:
        return _error_response(request_id, -32001, "resource not found"), True
    return _result_response(request_id, {"resource": enrich_resource_row(detail, state)}), True


def _handle_passes(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import get_pass_hierarchy

    actions = state.adapter.get_root_actions()
    tree = get_pass_hierarchy(actions, state.structured_file)

    return _result_response(request_id, {"tree": tree}), True


def _rewrite_pass_suffix(detail: dict[str, Any]) -> None:
    """Rewrite pass name suffix using actual target counts (matching GUI)."""
    name = detail.get("name", "")
    n_color = len(detail.get("color_targets") or [])
    has_depth = detail.get("depth_target") is not None
    if name.startswith("Depth-only Pass"):
        if n_color > 0:
            m = re.search(r"#(\d+)", name)
            num = m.group(1) if m else "0"
            parts: list[str] = [f"{n_color} Targets"]
            if has_depth:
                parts.append("Depth")
            detail["name"] = f"Colour Pass #{num} ({' + '.join(parts)})"
        return
    if not name.startswith("Colour Pass"):
        return
    if n_color == 0 and has_depth:
        m = re.search(r"#(\d+)", name)
        num = m.group(1) if m else "0"
        detail["name"] = f"Depth-only Pass #{num}"
        return
    parts2: list[str] = []
    if n_color:
        parts2.append(f"{n_color} Targets")
    if has_depth:
        parts2.append("Depth")
    suffix = f" ({' + '.join(parts2)})" if parts2 else ""
    prefix = name.split("(")[0].rstrip() if "(" in name else name
    detail["name"] = f"{prefix}{suffix}"


def _handle_pass(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import get_pass_detail

    identifier: int | str
    if "index" in params:
        try:
            identifier = int(params["index"])
        except (TypeError, ValueError):
            return _error_response(request_id, -32602, "index must be an integer"), True
    elif "name" in params:
        name = str(params["name"])
        if state.vfs_tree and name in state.vfs_tree.pass_name_map:
            name = state.vfs_tree.pass_name_map[name]
        identifier = name
    else:
        return _error_response(request_id, -32602, "missing index or name"), True
    actions = state.adapter.get_root_actions()
    detail = get_pass_detail(actions, state.structured_file, identifier)
    if detail is None:
        return _error_response(request_id, -32001, "pass not found"), True
    err = _seek_replay(state, detail["begin_eid"])
    if err is None:
        pipe = state.adapter.get_pipeline_state()
        detail["color_targets"] = [
            _enrich_target(int(t.resource), state, t)
            for t in pipe.GetOutputTargets()
            if int(t.resource) != 0
        ]
        depth_attach = pipe.GetDepthTarget()
        depth_id = int(depth_attach.resource)
        detail["depth_target"] = (
            _enrich_target(depth_id, state, depth_attach) if depth_id != 0 else None
        )
    else:
        detail["color_targets"] = []
        detail["depth_target"] = None
    # root-cause-5：attach 对象无 loadOp/clear* 属性 → 用结构化文件补
    # （vkCreateRenderPass load/store ops + vkCmdBeginRenderPass clear values）
    _fill_sf_attach_info(state, detail, identifier)
    _rewrite_pass_suffix(detail)
    return _result_response(request_id, detail), True


def _enrich_target(
    rid: int, state: DaemonState, attach: Any | None = None
) -> dict[str, Any]:
    """Build an enriched attachment dict for a render target resource ID.

    ``attach`` is the RenderDoc attachment object (e.g. BoundFramebufferAttachment);
    its loadOp/storeOp/clear* attributes are copied when present. ``None`` values
    keep the output stable across API versions that lack these attributes.
    """
    entry: dict[str, Any] = {"id": rid}
    name = state.res_names.get(rid, "")
    if name:
        entry["name"] = name
    tex = state.tex_map.get(rid)
    if tex is not None:
        fmt = getattr(tex, "format", None)
        if fmt and hasattr(fmt, "Name"):
            entry["format"] = fmt.Name()
        entry["width"] = tex.width
        entry["height"] = tex.height
    if attach is None:
        return entry
    entry["load_op"] = _enum_name(getattr(attach, "loadOp", None))
    entry["store_op"] = _enum_name(getattr(attach, "storeOp", None))
    clear_color = getattr(attach, "clearColor", None)
    if isinstance(clear_color, (list, tuple)):
        entry["clear_color"] = [float(v) for v in clear_color[:4]]
    elif clear_color is not None and hasattr(clear_color, "r"):
        entry["clear_color"] = [clear_color.r, clear_color.g, clear_color.b, clear_color.a]
    else:
        entry["clear_color"] = None
    entry["clear_depth"] = getattr(attach, "clearDepth", None)
    entry["clear_stencil"] = getattr(attach, "clearStencil", None)
    return entry


# ---------------------------------------------------------------------------
# root-cause-5：renderpass load/store/clear 值（结构化文件来源，无需重放）
#
# BoundFramebufferAttachment（GetOutputTargets 返回项）没有 loadOp/storeOp/
# clear* 属性（RenderDoc API 限制，Vulkan 下同样拿不到）。可靠来源是
# capture 的结构化文件（OpenFile 时本地解析，本地/远程重放都可用）：
#   - vkCreateRenderPass.pAttachments[i].loadOp/storeOp/stencilLoadOp/stencilStoreOp
#   - vkCmdBeginRenderPass.RenderPassBegin.pClearValues[i]（color/depthStencil）
#   - 映射：pass 附件 rid 列表（color_targets[].id + depth_target.id）==
#     vkCreateFramebuffer.pAttachments（AsResourceId，顺序一致）
# ---------------------------------------------------------------------------
from typing import Any as _Any

# sf 对象不可哈希（mock dataclass 与部分 swig 对象），用 id(sf) 做键并持有
# 引用防 id 复用；同一 daemon 会话内 structured file 固定不变。
_SF_RP_INDEX_CACHE: dict[int, tuple[_Any, tuple]] = {}


def _sf_find(ch, name: str, depth: int = 0):
    """Recursively find a child chunk by name. Defensive; returns None on any error."""
    if ch is None or depth > 8:
        return None
    try:
        if ch.name == name:
            return ch
        for i in range(ch.NumChildren()):
            found = _sf_find(ch.GetChild(i), name, depth + 1)
            if found is not None:
                return found
    except Exception:
        pass
    return None


def _sf_text(ch) -> str | None:
    if ch is None:
        return None
    try:
        return ch.AsString()
    except Exception:
        return None


def _vk_attach_op(s) -> str | None:
    """VK_ATTACHMENT_LOAD_OP_CLEAR → 'Clear'; ..._LOAD → 'Load'; ..._DONT_CARE → 'DontCare'."""
    if not isinstance(s, str):
        return None
    u = s.upper()
    if u.endswith("_DONT_CARE"):
        return "DontCare"
    for op in ("CLEAR", "LOAD", "STORE", "RESOLVE", "NONE", "UNDEFINED"):
        if u.endswith("_" + op):
            return op.title()
    return None


def _sf_renderpass_index(sf) -> tuple[dict, dict, list]:
    """Build renderpass attachment info from the structured file.

    Returns ``(fbs, rps, begins)``:
      - ``fbs``:    {framebuffer rid: [attachment resource ids]}  -- vkCreateFramebuffer
      - ``rps``:    {renderpass rid: {slot: {"load_op", "store_op",
                    "stencil_load_op", "stencil_store_op"}}}      -- vkCreateRenderPass
      - ``begins``: [(fb_rid, rp_rid, [clear dicts per slot])]    -- vkCmdBeginRenderPass,
                    in execution order; clear dicts carry "clear_color"/"clear_depth"/
                    "clear_stencil" when present.
    """
    _key = id(sf)
    _hit = _SF_RP_INDEX_CACHE.get(_key)
    if _hit is not None and _hit[0] is sf:
        return _hit[1]
    fbs: dict[int, list[int]] = {}
    rps: dict[int, dict[int, dict[str, str | None]]] = {}
    begins: list[tuple[int, int, list[dict]]] = []
    view_to_image: dict[int, int] = {}
    # 第一遍：view → image 映射（vkCreateFramebuffer 的 pAttachments 是 view 句柄）
    for chunk in sf.chunks:
        try:
            if chunk.name != "vkCreateImageView":
                continue
            ci = _sf_find(chunk, "CreateInfo")
            img = _sf_find(ci, "image") if ci else None
            view = _sf_find(chunk, "View")
            vrid = int(view.AsResourceId()) if view is not None else -1
            irid = int(img.AsResourceId()) if img is not None else -1
            if vrid >= 0 and irid >= 0:
                view_to_image[vrid] = irid
        except Exception:
            pass
    for chunk in sf.chunks:
        try:
            name = chunk.name
        except Exception:
            continue
        if name == "vkCreateFramebuffer":
            try:
                ci = _sf_find(chunk, "CreateInfo")
                fb = _sf_find(chunk, "Framebuffer")
                atts = _sf_find(ci, "pAttachments") if ci else None
                fb_rid = int(fb.AsResourceId()) if fb is not None else -1
                # pAttachments 是 VkImageView 句柄 → 映射到 image rid
                #（GetOutputTargets/pass_details 的附件 id 是 image rid）
                atts_rids = (
                    [
                        view_to_image.get(int(atts.GetChild(i).AsResourceId()), int(atts.GetChild(i).AsResourceId()))
                        for i in range(atts.NumChildren())
                    ]
                    if atts is not None
                    else []
                )
                if fb_rid >= 0:
                    fbs[fb_rid] = atts_rids
            except Exception:
                pass
        elif name == "vkCreateRenderPass":
            try:
                ci = _sf_find(chunk, "CreateInfo")
                rp = _sf_find(chunk, "RenderPass")
                atts = _sf_find(ci, "pAttachments") if ci else None
                rp_rid = int(rp.AsResourceId()) if rp is not None else -1
                ops: dict[int, dict[str, str | None]] = {}
                if atts is not None:
                    for i in range(atts.NumChildren()):
                        a = atts.GetChild(i)
                        ops[i] = {
                            "load_op": _vk_attach_op(_sf_text(_sf_find(a, "loadOp"))),
                            "store_op": _vk_attach_op(_sf_text(_sf_find(a, "storeOp"))),
                            "stencil_load_op": _vk_attach_op(_sf_text(_sf_find(a, "stencilLoadOp"))),
                            "stencil_store_op": _vk_attach_op(_sf_text(_sf_find(a, "stencilStoreOp"))),
                        }
                if rp_rid >= 0:
                    rps[rp_rid] = ops
            except Exception:
                pass
        elif name == "vkCmdBeginRenderPass":
            try:
                rb = _sf_find(chunk, "RenderPassBegin")
                fb = _sf_find(rb, "framebuffer") if rb else None
                rp = _sf_find(rb, "renderPass") if rb else None
                pv = _sf_find(rb, "pClearValues") if rb else None
                fb_rid = int(fb.AsResourceId()) if fb is not None else -1
                rp_rid = int(rp.AsResourceId()) if rp is not None else -1
                clears: list[dict] = []
                if pv is not None:
                    for i in range(pv.NumChildren()):
                        el = pv.GetChild(i)
                        color = _sf_find(el, "color")
                        ds = _sf_find(el, "depthStencil")
                        entry: dict = {}
                        if color is not None:
                            f32 = _sf_find(color, "float32")
                            if f32 is not None:
                                entry["clear_color"] = [
                                    float(f32.GetChild(k).AsFloat())
                                    for k in range(min(4, f32.NumChildren()))
                                ]
                        if ds is not None:
                            d = _sf_find(ds, "depth")
                            s = _sf_find(ds, "stencil")
                            if d is not None:
                                try:
                                    entry["clear_depth"] = float(d.AsFloat())
                                except Exception:
                                    pass
                            if s is not None:
                                try:
                                    entry["clear_stencil"] = float(s.AsFloat())
                                except Exception:
                                    pass
                        clears.append(entry)
                begins.append((fb_rid, rp_rid, clears))
            except Exception:
                pass
    _idx = (fbs, rps, begins)
    _SF_RP_INDEX_CACHE[_key] = (sf, _idx)
    return fbs, rps, begins


def _fill_sf_attach_info(state: DaemonState, detail: dict, identifier: int | str) -> None:
    """Fill load_op/store_op/clear_* from the structured file into pass detail.

    Only overwrites fields left None by ``_enrich_target`` (attach attributes
    are absent in every RenderDoc API); keeps analyst/other sources' values.
    """
    sf = state.structured_file
    if sf is None:
        return
    color_targets = detail.get("color_targets") or []
    depth_target = detail.get("depth_target")
    if not color_targets and not depth_target:
        return
    try:
        fbs, rps, begins = _sf_renderpass_index(sf)
    except Exception:
        return
    att_rids = [c.get("id") for c in color_targets if c.get("id") is not None]
    if depth_target and depth_target.get("id"):
        att_rids.append(depth_target["id"])
    if not att_rids:
        return
    fb_rid = None
    for rid, atts in fbs.items():
        if atts == att_rids:
            fb_rid = rid
            break
    if fb_rid is None:
        return
    fb_begins = [b for b in begins if b[0] == fb_rid]
    if not fb_begins:
        return
    # 同 fb 多 pass 复用（ping-pong 同 RT 集合）时按 pass 序次取对应 begin；
    # 唯一匹配（最常见）直接取第一个。
    begin = fb_begins[0]
    if len(fb_begins) > 1:
        try:
            from rdc.services.query_service import _pass_list_with_fallback

            actions = state.adapter.get_root_actions()
            passes = _pass_list_with_fallback(actions, sf)
            cur = next(
                i
                for i, p in enumerate(passes)
                if p.get("begin_eid") == detail.get("begin_eid")
                and p.get("end_eid") == detail.get("end_eid")
            )
            # begins 与 passes 同为执行序（graphics pass ↔ vkCmdBeginRenderPass
            # 按序 1:1）。当前 pass 的 begin 在 begins 中的位置 = 它前面「有
            # begin」的 pass 数（纯 compute 等无 vkCmdBeginRenderPass 的 pass
            # 不占位）。同 fb 第 use 次复用 → 取 fb_begins[use]：按序 pop，而
            # 不是用全局 pass 序号 cur 直接做 fb_begins 下标（复用时会把所有
            # use 都塌到最后一个 begin）。
            def _has_begin(p: dict) -> bool:
                return not (
                    p.get("draws", 0) == 0
                    and p.get("dispatches", 0) > 0
                    and (p.get("copies", 0) or 0) == 0
                    and (p.get("clears", 0) or 0) == 0
                )

            pb = sum(1 for p in passes[:cur] if _has_begin(p))
            use = sum(1 for pos, b in enumerate(begins) if b[0] == fb_rid and pos < pb)
            begin = fb_begins[min(use, len(fb_begins) - 1)]
        except Exception:
            begin = fb_begins[0]
    _rp_rid, clears = begin[1], begin[2]
    ops = rps.get(_rp_rid, {})
    for i, ct in enumerate(color_targets):
        op = ops.get(i, {})
        clear = clears[i] if i < len(clears) else {}
        if ct.get("load_op") is None:
            ct["load_op"] = op.get("load_op")
        if ct.get("store_op") is None:
            ct["store_op"] = op.get("store_op")
        if ct.get("clear_color") is None and "clear_color" in clear:
            ct["clear_color"] = clear["clear_color"]
    if depth_target:
        dslot = len(color_targets)
        op = ops.get(dslot, {})
        clear = clears[dslot] if dslot < len(clears) else {}
        if depth_target.get("load_op") is None:
            depth_target["load_op"] = op.get("load_op")
        if depth_target.get("store_op") is None:
            depth_target["store_op"] = op.get("store_op")
        if depth_target.get("clear_depth") is None and "clear_depth" in clear:
            depth_target["clear_depth"] = clear["clear_depth"]
        if depth_target.get("clear_stencil") is None and "clear_stencil" in clear:
            depth_target["clear_stencil"] = clear["clear_stencil"]


def _handle_log(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    controller = state.adapter.controller
    level_filter = params.get("level")
    if level_filter is not None:
        level_filter = str(level_filter).upper()
        if level_filter not in _VALID_LOG_LEVELS:
            return _error_response(request_id, -32602, f"invalid level: {level_filter}"), True
    eid_filter = params.get("eid")
    if eid_filter is not None:
        try:
            eid_filter = int(eid_filter)
        except (TypeError, ValueError):
            return _error_response(request_id, -32602, "eid must be an integer"), True
    if state._debug_messages_cache is None:
        raw = controller.GetDebugMessages() if hasattr(controller, "GetDebugMessages") else []
        state._debug_messages_cache = list(raw)
    msgs = state._debug_messages_cache
    log_rows: list[dict[str, Any]] = []
    for m in msgs:
        lvl = _LOG_SEVERITY_MAP.get(int(m.severity), "UNKNOWN")
        if level_filter and lvl != level_filter:
            continue
        raw_eid = int(m.eventId)
        if eid_filter is not None and raw_eid != eid_filter:
            continue
        log_rows.append({"level": lvl, "eid": raw_eid, "message": m.description})
    return _result_response(request_id, {"messages": log_rows}), True


def _handle_info(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import aggregate_stats

    flat = _get_flat_actions(state)
    stats = aggregate_stats(flat)
    result: dict[str, Any] = {
        "Capture": state.capture,
        "API": state.api_name,
        "Events": len(flat),
        "Draw Calls": (
            f"{stats.total_draws} "
            f"({stats.indexed_draws} indexed, "
            f"{stats.non_indexed_draws} non-indexed, "
            f"{stats.dispatches} dispatches)"
        ),
        "Clears": stats.clears,
        "Copies": stats.copies,
    }
    if state.cap is not None:
        result["has_callstacks"] = state.cap.HasCallstacks()
        result["machine_ident"] = state.cap.RecordedMachineIdent()
        result["timestamp_base"] = state.cap.TimestampBase()
    return _result_response(request_id, result), True


def _handle_stats(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import aggregate_stats, get_top_draws

    flat = _get_flat_actions(state)
    stats = aggregate_stats(flat)
    top = get_top_draws(flat, limit=3)

    # Find a representative draw EID per pass for RT enrichment
    from rdc.services.query_service import _DRAWCALL, _MESHDRAW

    pass_first_draw: dict[str, int] = {}
    for a in flat:
        if (a.flags & (_DRAWCALL | _MESHDRAW)) and a.pass_name not in pass_first_draw:
            pass_first_draw[a.pass_name] = a.eid
    for ps in stats.per_pass:
        draw_eid = pass_first_draw.get(ps.name)
        if draw_eid is None:
            continue
        err = _seek_replay(state, draw_eid)
        if err:
            continue
        try:
            pipe = state.adapter.get_pipeline_state()
            targets = pipe.GetOutputTargets()
            non_null = [t for t in targets if int(t.resource) != 0]
            ps.attachments = len(non_null)
            depth_id = int(pipe.GetDepthTarget().resource)
            if depth_id != 0:
                ps.attachments += 1
            if non_null:
                rid = int(non_null[0].resource)
                tex = state.tex_map.get(rid)
                if tex is not None:
                    ps.rt_w = tex.width
                    ps.rt_h = tex.height
        except Exception:
            pass  # fallback to defaults (0)

    # Restore replay head to user's position
    if state.current_eid != 0:
        _seek_replay(state, state.current_eid)

    per_pass = [
        {
            "name": ps.name,
            "draws": ps.draws,
            "dispatches": ps.dispatches,
            "triangles": ps.triangles,
            "rt_w": ps.rt_w or "-",
            "rt_h": ps.rt_h or "-",
            "attachments": ps.attachments,
        }
        for ps in stats.per_pass
    ]
    top_draws = [
        {
            "eid": a.eid,
            "marker": a.parent_marker,
            "triangles": (a.num_indices // 3) * a.num_instances,
        }
        for a in top
    ]

    # Largest resources by byte size
    largest: list[dict[str, Any]] = []
    for rid, res in state.res_rid_map.items():
        size = getattr(res, "byteSize", 0)
        if size <= 0:
            continue
        t = getattr(res, "type", None)
        type_name = getattr(t, "name", str(t)) if t is not None else ""
        fmt = "-"
        tex = state.tex_map.get(rid)
        if tex is not None:
            f = getattr(tex, "format", None)
            fmt = f.Name() if f and hasattr(f, "Name") else str(f) if f else "-"
        largest.append(
            {
                "id": rid,
                "name": getattr(res, "name", ""),
                "type": type_name,
                "size": size,
                "format": fmt,
            }
        )
    largest.sort(key=lambda r: r["size"], reverse=True)
    largest = largest[:5]

    return _result_response(
        request_id,
        {
            "per_pass": per_pass,
            "top_draws": top_draws,
            "largest_resources": largest,
            "event_count": state.max_eid,
        },
    ), True


def _handle_events(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import filter_by_pattern, filter_by_type

    flat = _get_flat_actions(state)
    event_type = params.get("type")
    if event_type:
        flat = filter_by_type(flat, event_type)
    pattern = params.get("filter")
    if pattern:
        flat = filter_by_pattern(flat, pattern)
    eid_range = params.get("range")
    if eid_range and ":" in str(eid_range):
        parts = str(eid_range).split(":", 1)
        lo = int(parts[0]) if parts[0] else 0
        hi = int(parts[1]) if parts[1] else 999999999
        flat = [a for a in flat if lo <= a.eid <= hi]
    limit = params.get("limit")
    if limit is not None:
        flat = flat[: int(limit)]
    events = [{"eid": a.eid, "type": _action_type_str(a.flags), "name": a.name} for a in flat]
    return _result_response(request_id, {"events": events}), True


def _handle_draws(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import (
        aggregate_stats,
        filter_by_pass,
        filter_by_type,
        pass_name_for_eid,
    )

    all_flat = _get_flat_actions(state)
    passes = state.vfs_tree.pass_list if state.vfs_tree else []
    pass_name = params.get("pass")
    if pass_name:
        _actions = state.adapter.get_root_actions()
        _sf = state.structured_file
        all_flat = filter_by_pass(all_flat, pass_name, actions=_actions, sf=_sf)
    stats = aggregate_stats(all_flat)
    flat = filter_by_type(all_flat, "draw")
    sort_field = params.get("sort")
    if sort_field == "triangles":
        flat.sort(key=lambda a: (a.num_indices // 3) * a.num_instances, reverse=True)
    limit = params.get("limit")
    if limit is not None:
        flat = flat[: int(limit)]
    draws = [
        {
            "eid": a.eid,
            "type": _action_type_str(a.flags),
            "triangles": (a.num_indices // 3) * a.num_instances,
            "instances": a.num_instances,
            "pass": pass_name_for_eid(a.eid, passes),
            "marker": a.parent_marker,
        }
        for a in flat
    ]
    summary = (
        f"{stats.total_draws} draw calls "
        f"({stats.indexed_draws} indexed, "
        f"{stats.dispatches} dispatches, "
        f"{stats.clears} clears)"
    )
    return _result_response(request_id, {"draws": draws, "summary": summary}), True


def _extract_sd_value(child: Any) -> str:
    """Extract typed value from SDObject based on basetype."""
    sd_type = getattr(child, "type", None)
    if sd_type is None:
        return child.AsString() if hasattr(child, "AsString") else "-"
    basetype = getattr(sd_type, "basetype", -1)
    data = getattr(child, "data", None)
    if data is None:
        return child.AsString() if hasattr(child, "AsString") else "-"
    basic = getattr(data, "basic", None)
    if basetype == 7:  # UnsignedInteger
        return str(getattr(basic, "u", 0)) if basic else "-"
    if basetype == 8:  # SignedInteger
        return str(getattr(basic, "i", 0)) if basic else "-"
    if basetype == 9:  # Float
        return str(getattr(basic, "d", 0.0)) if basic else "-"
    if basetype == 10:  # Boolean
        return str(bool(getattr(basic, "b", False))) if basic else "-"
    if basetype == 12:  # Resource
        return str(int(getattr(basic, "id", 0))) if basic else "-"
    if basetype in (5, 6):  # String, Enum
        s = getattr(data, "str", "")
        return s if s else (child.AsString() if hasattr(child, "AsString") else "-")
    return child.AsString() if hasattr(child, "AsString") else "-"


def _handle_event(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    eid = params.get("eid")
    if eid is None:
        return _error_response(request_id, -32602, "missing eid parameter"), True
    from rdc.services.query_service import find_action_by_eid

    eid = int(eid)
    action = find_action_by_eid(state.adapter.get_root_actions(), eid)
    if action is None:
        return (
            _error_response(request_id, -32002, f"eid {eid} out of range (max: {state.max_eid})"),
            True,
        )
    sf = state.structured_file
    api_call = "-"
    params_dict: dict[str, Any] = {}
    if action.events and sf and hasattr(sf, "chunks"):
        for evt in action.events:
            idx = evt.chunkIndex
            if 0 <= idx < len(sf.chunks):
                chunk = sf.chunks[idx]
                api_call = chunk.name
                if hasattr(chunk, "NumChildren"):
                    for ci in range(chunk.NumChildren()):
                        child = chunk.GetChild(ci)
                        val = _extract_sd_value(child)
                        params_dict[child.name] = val
                else:
                    for child in chunk.children:
                        val = child.data.basic.value if child.data and child.data.basic else "-"
                        params_dict[child.name] = val
    result: dict[str, Any] = {"EID": eid, "API Call": api_call}
    if params_dict:
        param_str = chr(10).join(f"  {k:<20}{v}" for k, v in params_dict.items())
        result["Parameters"] = chr(10) + param_str
    else:
        result["Parameters"] = "-"
    result["Duration"] = "-"
    return _result_response(request_id, result), True


def _handle_draw(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import find_action_by_eid, walk_actions

    eid = params.get("eid")
    if eid is None:
        eid = state.current_eid
    eid = int(eid)
    root_actions = state.adapter.get_root_actions()
    action = find_action_by_eid(root_actions, eid)
    if action is None:
        return (
            _error_response(request_id, -32002, f"eid {eid} out of range (max: {state.max_eid})"),
            True,
        )
    sf = state.structured_file
    flat = walk_actions(root_actions, sf)
    flat_match = [a for a in flat if a.eid == eid]
    name = action.GetName(sf) if sf else getattr(action, "_name", "-")
    marker = flat_match[0].parent_marker if flat_match else "-"
    tris = (action.numIndices // 3) * max(action.numInstances, 1)
    return _result_response(
        request_id,
        {
            "Event": eid,
            "Type": name,
            "Marker": marker,
            "Triangles": tris,
            "Instances": max(action.numInstances, 1),
        },
    ), True


def _handle_search(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    pattern = str(params.get("pattern", ""))
    if not pattern:
        return _error_response(request_id, -32602, "missing pattern"), True
    if len(pattern) > 500:
        return _error_response(request_id, -32602, "pattern too long (max 500)"), True
    stage_filter = params.get("stage")
    if stage_filter is not None:
        stage_filter = str(stage_filter).lower()
    case_sensitive = bool(params.get("case_sensitive", False))
    limit = max(1, int(params.get("limit", 200)))
    context_lines = int(params.get("context", 0))
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern, flags)
    except re.error as exc:
        return _error_response(request_id, -32602, f"invalid regex: {exc}"), True
    _build_shader_cache(state)
    matches: list[dict[str, Any]] = []
    truncated = False
    for sid, text in state.disasm_cache.items():
        meta = state.shader_meta.get(sid, {})
        shader_stages: list[str] = meta.get("stages", [])
        if stage_filter and stage_filter not in shader_stages:
            continue
        lines = text.splitlines()
        for lineno, line in enumerate(lines):
            if compiled.search(line):
                ctx_before = lines[max(0, lineno - context_lines) : lineno]
                ctx_after = lines[lineno + 1 : lineno + 1 + context_lines]
                matches.append(
                    {
                        "shader": sid,
                        "stages": shader_stages,
                        "first_eid": meta.get("first_eid", 0),
                        "line": lineno + 1,
                        "text": line,
                        "context_before": ctx_before,
                        "context_after": ctx_after,
                    }
                )
                if len(matches) >= limit:
                    truncated = True
                    break
        if truncated:
            break
    return _result_response(request_id, {"matches": matches, "truncated": truncated}), True


def _handle_pass_deps(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    from rdc.services.query_service import _build_pass_list, build_pass_deps

    actions = state.adapter.get_root_actions()
    passes = _build_pass_list(actions, state.structured_file)
    usage_data: dict[int, list[Any]] = {}
    for resid, rid_obj in state.res_rid_map.items():
        usage_data[resid] = state.adapter.controller.GetUsage(rid_obj.resourceId)
    result = build_pass_deps(passes, usage_data)
    return _result_response(request_id, result), True


def _handle_pass_attachment(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    """Return attachment info for a specific pass attachment."""
    if state.adapter is None:
        return _error_response(request_id, -32002, "no replay loaded"), True
    from rdc.services.query_service import get_pass_detail

    name = str(params.get("name", ""))
    attachment = str(params.get("attachment", ""))
    if state.vfs_tree and name in state.vfs_tree.pass_name_map:
        name = state.vfs_tree.pass_name_map[name]

    actions = state.adapter.get_root_actions()
    detail = get_pass_detail(actions, state.structured_file, name)
    if detail is None:
        return _error_response(request_id, -32001, f"pass not found: {name}"), True

    err = _seek_replay(state, detail["begin_eid"])
    if err:
        return _error_response(request_id, -32002, err), True
    pipe = state.adapter.get_pipeline_state()

    if attachment == "depth":
        depth_id = int(pipe.GetDepthTarget().resource)
        if depth_id == 0:
            return _error_response(request_id, -32001, "no depth target"), True
        return _result_response(
            request_id, {"pass": name, "attachment": "depth", "resource_id": depth_id}
        ), True

    if attachment.startswith("color"):
        try:
            idx = int(attachment[5:])
        except ValueError:
            return _error_response(request_id, -32602, f"invalid attachment: {attachment}"), True
        all_targets = pipe.GetOutputTargets()
        if idx < 0 or idx >= len(all_targets) or int(all_targets[idx].resource) == 0:
            return _error_response(request_id, -32001, f"color target {idx} not found"), True
        return _result_response(
            request_id,
            {"pass": name, "attachment": attachment, "resource_id": int(all_targets[idx].resource)},
        ), True

    return _error_response(request_id, -32001, f"unknown attachment: {attachment}"), True


def _handle_preload(
    request_id: int, params: dict[str, Any], state: DaemonState
) -> tuple[dict[str, Any], bool]:
    assert state.adapter is not None
    _build_shader_cache(state)
    return _result_response(
        request_id,
        {"done": True, "shaders": len(state.disasm_cache)},
    ), True


HANDLERS: dict[str, Handler] = {
    "shader_map": _handle_shader_map,
    "pipeline": _handle_pipeline,
    "bindings": _handle_bindings,
    "shader": _handle_shader,
    "shaders": _handle_shaders,
    "resources": _handle_resources,
    "resource": _handle_resource,
    "passes": _handle_passes,
    "pass": _handle_pass,
    "pass_deps": _handle_pass_deps,
    "pass_attachment": _handle_pass_attachment,
    "log": _handle_log,
    "info": _handle_info,
    "stats": _handle_stats,
    "events": _handle_events,
    "draws": _handle_draws,
    "event": _handle_event,
    "draw": _handle_draw,
    "search": _handle_search,
    "shaders_preload": _handle_preload,
}
