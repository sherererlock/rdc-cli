"""Tests for remote replay infrastructure in daemon_server.py."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rdc.daemon_server import (
    _REMOTE_TIMEOUT_MS,
    DaemonState,
    _load_remote_replay,
    _load_replay,
    _raise_remote_timeout,
    _start_ping_thread,
    _stop_ping_thread,
)


def _fake_setting(value: int) -> Any:
    """A minimal stand-in for the SDObject SetConfigSetting() returns."""
    return SimpleNamespace(data=SimpleNamespace(basic=SimpleNamespace(u=value)))


def _make_mock_rd(
    *,
    connect_result: int = 0,
    open_capture_result: int = 0,
    open_file_result: int = 0,
    copy_to_remote_path: str = "/tmp/RenderDoc/frame.rdc",
) -> tuple[MagicMock, MagicMock]:
    """Build mock rd module and remote server."""
    mock_remote = MagicMock()
    mock_remote.OpenCapture.return_value = (open_capture_result, MagicMock())
    mock_remote.CopyCaptureToRemote.return_value = copy_to_remote_path
    mock_remote.CopyCaptureFromRemote.return_value = None
    mock_remote.Ping.return_value = None
    mock_remote.CloseCapture.return_value = None
    mock_remote.ShutdownConnection.return_value = None

    mock_cap = MagicMock()
    mock_cap.OpenFile.return_value = open_file_result
    mock_cap.GetStructuredData.return_value = MagicMock()

    rd = MagicMock()
    rd.ResultCode.Succeeded = 0
    rd.InitialiseReplay.return_value = None
    rd.CreateRemoteServerConnection.return_value = (connect_result, mock_remote)
    rd.RemoteServer.NoPreference = 0
    rd.ReplayOptions.return_value = MagicMock()
    rd.OpenCaptureFile.return_value = mock_cap
    rd.GetVersionString.return_value = "1.41"

    return rd, mock_remote


class TestLoadRemoteReplay:
    def test_no_renderdoc_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: None)
        state = DaemonState(capture="/tmp/frame.rdc", current_eid=0, token="tok")
        err = _load_remote_replay(state, "host:39920")
        assert err is not None
        assert "failed to import renderdoc module" in err

    def test_init_replay_fails_continues(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """B39: InitialiseReplay failure does not abort remote replay."""
        rd, _remote = _make_mock_rd()
        rd.InitialiseReplay.side_effect = RuntimeError("no local GPU")
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")

        with patch("rdc.daemon_server._init_adapter_state"):
            err = _load_remote_replay(state, "host:39920")

        assert err is None

    def test_init_replay_failure_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """B39: InitialiseReplay failure emits a warning log."""
        rd, _remote = _make_mock_rd()
        rd.InitialiseReplay.side_effect = RuntimeError("no local GPU")
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")

        with caplog.at_level(logging.WARNING, logger="rdc.daemon"):
            with patch("rdc.daemon_server._init_adapter_state"):
                _load_remote_replay(state, "host:39920")

        assert any("InitialiseReplay" in r.message for r in caplog.records)

    def test_connection_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rd, _remote = _make_mock_rd(connect_result=1)
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)
        state = DaemonState(capture="/tmp/frame.rdc", current_eid=0, token="tok")
        err = _load_remote_replay(state, "host:39920")
        assert err is not None
        assert "remote connection failed" in err

    def test_local_file_uploads(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        rd, mock_remote = _make_mock_rd()
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")

        with patch("rdc.daemon_server._init_adapter_state"):
            err = _load_remote_replay(state, "host:39920")

        assert err is None
        args = mock_remote.CopyCaptureToRemote.call_args[0]
        assert args[0] == str(local_capture)
        assert callable(args[1])
        assert state.local_capture_path == str(local_capture)

    def test_remote_path_downloads(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        rd, mock_remote = _make_mock_rd()
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        state = DaemonState(capture="/remote/captures/frame.rdc", current_eid=0, token="tok12345")

        with patch("rdc.daemon_server._init_adapter_state"):
            err = _load_remote_replay(state, "host:39920")

        assert err is None
        mock_remote.CopyCaptureFromRemote.assert_called_once()
        assert state.local_capture_path != ""
        assert "rdc-remote-" in state.local_capture_path
        assert state.local_capture_is_temp

    def test_copy_from_remote_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rd, mock_remote = _make_mock_rd()
        mock_remote.CopyCaptureFromRemote.side_effect = OSError("network error")
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        state = DaemonState(capture="/remote/captures/frame.rdc", current_eid=0, token="tok12345")
        err = _load_remote_replay(state, "host:39920")
        assert err is not None
        assert "at step 'download capture'" in err
        assert mock_remote.ShutdownConnection.call_count == 1

    def test_open_capture_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        rd, mock_remote = _make_mock_rd(open_capture_result=1)
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")
        err = _load_remote_replay(state, "host:39920")
        assert err is not None
        assert "remote OpenCapture failed" in err
        assert mock_remote.ShutdownConnection.call_count == 1

    def test_local_openfile_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        rd, mock_remote = _make_mock_rd(open_file_result=1)
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")
        err = _load_remote_replay(state, "host:39920")
        assert err is not None
        assert "local OpenFile (metadata) failed" in err
        assert mock_remote.CloseCapture.call_count == 1
        assert mock_remote.ShutdownConnection.call_count == 1

    def test_success_sets_state_fields(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rd, _mock_remote = _make_mock_rd()
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")

        with patch("rdc.daemon_server._init_adapter_state"):
            err = _load_remote_replay(state, "host:39920")

        assert err is None
        assert state.is_remote is True
        assert state.remote_url == "host:39920"
        assert state.adapter is not None
        assert state.cap is not None
        assert state._ping_thread is not None

    def test_does_not_force_a_local_gpu_onto_remote_opencapture(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression: replay executes on the remote device, so forceGPUVendor/
        forceGPUDeviceID must not be set from this machine's GetAvailableGPUs()."""
        rd, mock_remote = _make_mock_rd()
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        def _boom_match(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("_match_capture_gpu must not run for remote replay")

        monkeypatch.setattr("rdc.daemon_server._match_capture_gpu", _boom_match)

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")

        with patch("rdc.daemon_server._init_adapter_state"):
            err = _load_remote_replay(state, "host:39920")

        assert err is None
        # _boom_match would have raised above if the remote path still probed a local
        # GPU. The only remaining local OpenCaptureFile call is "open local metadata".
        assert rd.OpenCaptureFile.call_count == 1
        assert mock_remote.OpenCapture.call_count == 1

    def test_raises_remote_timeout_before_connecting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """RemoteServer.TimeoutMS defaults to 5000ms, shorter than the gaps between
        LogOpenProgress packets a slow/mobile replay host can leave while opening a large
        capture -- raise it before RENDERDOC_CreateRemoteServerConnection reads it."""
        rd, _mock_remote = _make_mock_rd()
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)
        setting = _fake_setting(5000)
        rd.SetConfigSetting.return_value = setting

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")

        with patch("rdc.daemon_server._init_adapter_state"):
            err = _load_remote_replay(state, "host:39920")

        assert err is None
        rd.SetConfigSetting.assert_called_once_with("RemoteServer.TimeoutMS")
        assert setting.data.basic.u == _REMOTE_TIMEOUT_MS


class TestRaiseRemoteTimeout:
    def test_raises_low_timeout(self) -> None:
        setting = _fake_setting(5000)
        rd = MagicMock()
        rd.SetConfigSetting.return_value = setting

        _raise_remote_timeout(rd)

        rd.SetConfigSetting.assert_called_once_with("RemoteServer.TimeoutMS")
        assert setting.data.basic.u == _REMOTE_TIMEOUT_MS

    def test_does_not_lower_an_existing_higher_value(self) -> None:
        setting = _fake_setting(120_000)
        rd = MagicMock()
        rd.SetConfigSetting.return_value = setting

        _raise_remote_timeout(rd)

        assert setting.data.basic.u == 120_000

    def test_missing_config_setting_is_non_fatal(self) -> None:
        rd = MagicMock()
        rd.SetConfigSetting.return_value = None

        _raise_remote_timeout(rd)  # must not raise

    def test_set_config_setting_unavailable_is_non_fatal(self) -> None:
        rd = MagicMock()
        rd.SetConfigSetting.side_effect = AttributeError("no such API on this renderdoc build")

        _raise_remote_timeout(rd)  # must not raise


class TestLoadRemoteReplayStepLabels:
    """T24 group C: step labels in remote replay setup failure messages."""

    def test_upload_capture_runtime_error_step_label(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rd, mock_remote = _make_mock_rd()
        mock_remote.CopyCaptureToRemote.side_effect = RuntimeError("upload bang")
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")

        err = _load_remote_replay(state, "host:39920")
        assert err is not None
        assert "at step 'upload capture'" in err
        assert "upload bang" in err
        assert mock_remote.ShutdownConnection.call_count == 1

    def test_upload_capture_os_error_step_label(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rd, mock_remote = _make_mock_rd()
        mock_remote.CopyCaptureToRemote.side_effect = OSError("disk full")
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")

        err = _load_remote_replay(state, "host:39920")
        assert err is not None
        assert "at step 'upload capture'" in err
        assert "disk full" in err

    def test_outer_catchall_labels_init_adapter_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Unexpected exception in _init_adapter_state is wrapped with step label + type."""
        rd, mock_remote = _make_mock_rd()
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)

        def _boom(_state: Any) -> None:
            raise ValueError("adapter bang")

        monkeypatch.setattr("rdc.daemon_server._init_adapter_state", _boom)

        local_capture = tmp_path / "frame.rdc"
        local_capture.write_bytes(b"\x00")
        state = DaemonState(capture=str(local_capture), current_eid=0, token="tok12345")

        err = _load_remote_replay(state, "host:39920")
        assert err is not None
        assert "at step 'init adapter state'" in err
        assert "ValueError" in err
        assert "adapter bang" in err
        assert mock_remote.ShutdownConnection.call_count == 1


class TestLoadReplayRegressionB39:
    """B39 regression: _load_replay must still fail on InitialiseReplay error."""

    def test_load_replay_still_fails_on_init_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rd = MagicMock()
        rd.InitialiseReplay.side_effect = RuntimeError("init boom")
        monkeypatch.setattr("rdc.discover.find_renderdoc", lambda: rd)
        state = DaemonState(capture="/tmp/frame.rdc", current_eid=0, token="tok")
        err = _load_replay(state)
        assert err is not None
        assert "InitialiseReplay failed" in err


class TestPingThread:
    def test_start_sets_state(self) -> None:
        state = DaemonState(capture="frame.rdc", current_eid=0, token="tok")
        state.remote = MagicMock()
        _start_ping_thread(state)
        try:
            assert isinstance(state._ping_stop, threading.Event)
            assert isinstance(state._ping_thread, threading.Thread)
            assert state._ping_thread.is_alive()
        finally:
            _stop_ping_thread(state)

    def test_ping_exception_exits_loop(self) -> None:
        state = DaemonState(capture="frame.rdc", current_eid=0, token="tok")
        state.remote = MagicMock()
        state.remote.Ping.side_effect = OSError("disconnect")
        _start_ping_thread(state)
        # Thread waits 3s then calls Ping() which raises; join with enough margin
        state._ping_thread.join(timeout=5.0)
        assert not state._ping_thread.is_alive()

    def test_stop_signals_and_joins(self) -> None:
        state = DaemonState(capture="frame.rdc", current_eid=0, token="tok")
        state.remote = MagicMock()
        _start_ping_thread(state)
        _stop_ping_thread(state)
        assert state._ping_stop.is_set()
        assert not state._ping_thread.is_alive()

    def test_stop_noop_when_not_started(self) -> None:
        state = DaemonState(capture="frame.rdc", current_eid=0, token="tok")
        _stop_ping_thread(state)  # should not raise
