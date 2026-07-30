from typing import cast

import pytest
import requests
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from src.api.client import main


class FakeConsole:
    def __init__(self):
        self.output = []

    def print(self, *args, **kwargs):
        self.output.append((args, kwargs))


class DummyStatus:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_startup_banner(monkeypatch):
    console = FakeConsole()

    monkeypatch.setattr("src.api.client.Console", lambda: console)
    monkeypatch.setattr("src.api.client.Prompt.ask", lambda *args, **kwargs: "exit")
    monkeypatch.setattr(
        "src.api.client.sys.exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )

    with pytest.raises(SystemExit):
        main()

    assert len(console.output) >= 2


def test_exit(monkeypatch):
    console = FakeConsole()

    monkeypatch.setattr("src.api.client.Console", lambda: console)
    monkeypatch.setattr("src.api.client.Prompt.ask", lambda *args, **kwargs: "exit")
    monkeypatch.setattr(
        "src.api.client.sys.exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0


def test_quit(monkeypatch):
    console = FakeConsole()

    monkeypatch.setattr("src.api.client.Console", lambda: console)
    monkeypatch.setattr("src.api.client.Prompt.ask", lambda *args, **kwargs: "quit")
    monkeypatch.setattr(
        "src.api.client.sys.exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )

    with pytest.raises(SystemExit):
        main()


def test_empty_prompt(monkeypatch):
    console = FakeConsole()

    answers = iter(["", "exit"])

    monkeypatch.setattr("src.api.client.Console", lambda: console)
    monkeypatch.setattr(
        "src.api.client.Prompt.ask", lambda *args, **kwargs: next(answers)
    )
    monkeypatch.setattr(
        "src.api.client.sys.exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )

    with pytest.raises(SystemExit):
        main()


def test_success(monkeypatch):
    console = FakeConsole()

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "status": 200,
                "generated_response": '{"thought_process" : "thought" , "answer" : "hello" , "sources" : ["source"]}',
            }

    answers = iter(["question", "exit"])

    monkeypatch.setattr("src.api.client.Console", lambda: console)
    monkeypatch.setattr(
        "src.api.client.Prompt.ask", lambda *args, **kwargs: next(answers)
    )
    monkeypatch.setattr(
        "src.api.client.requests.post", lambda *args, **kwargs: FakeResponse()
    )
    monkeypatch.setattr(
        "src.api.client.sys.exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )
    monkeypatch.setattr("src.api.client.Status", DummyStatus)

    with pytest.raises(SystemExit):
        main()

    panel = console.output[-1][0][0]
    response_panels = [
        args[0]
        for args, _ in console.output
        if args
        and isinstance(args[0], Panel)
        and args[0].title == "[bold blue]System Response[/bold blue]"
    ]
    assert len(response_panels) == 1
    panel = response_panels[0]
    assert isinstance(panel, Panel)
    group = panel.renderable
    assert isinstance(group, Group)
    thought_panel = cast(Panel, group.renderables[0])
    answer_panel = cast(Panel, group.renderables[1])
    sources_panel = cast(Panel, group.renderables[2])

    assert thought_panel.title == "[bold red]Thought Process[/bold red]"
    assert answer_panel.title == "[bold green]Answer[/bold green]"
    assert sources_panel.title == "[bold yellow]Sources[/bold yellow]"

    assert isinstance(thought_panel.renderable, Text)
    assert isinstance(answer_panel.renderable, Text)
    assert isinstance(sources_panel.renderable, Text)

    assert thought_panel.renderable.plain == "thought"
    assert answer_panel.renderable.plain == "hello"
    assert sources_panel.renderable.plain == "source"


def test_server_returns_error_response(monkeypatch):
    console = FakeConsole()

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "status": 500,
                "generated_response": "failure",
            }

    answers = iter(["question", "exit"])

    monkeypatch.setattr("src.api.client.Console", lambda: console)
    monkeypatch.setattr(
        "src.api.client.Prompt.ask", lambda *args, **kwargs: next(answers)
    )
    monkeypatch.setattr(
        "src.api.client.requests.post", lambda *args, **kwargs: FakeResponse()
    )
    monkeypatch.setattr(
        "src.api.client.sys.exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )
    monkeypatch.setattr("src.api.client.Status", DummyStatus)

    with pytest.raises(SystemExit):
        main()

    messages = [
        args[0] for args, _ in console.output if args and isinstance(args[0], str)
    ]
    assert any("Error" in msg for msg in messages)


def test_http_error(monkeypatch):
    console = FakeConsole()

    class FakeResponse:
        status_code = 503

    answers = iter(["question", "exit"])

    monkeypatch.setattr("src.api.client.Console", lambda: console)
    monkeypatch.setattr(
        "src.api.client.Prompt.ask", lambda *args, **kwargs: next(answers)
    )
    monkeypatch.setattr(
        "src.api.client.requests.post", lambda *args, **kwargs: FakeResponse()
    )
    monkeypatch.setattr(
        "src.api.client.sys.exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )
    monkeypatch.setattr("src.api.client.Status", DummyStatus)

    with pytest.raises(SystemExit):
        main()

    messages = [
        args[0] for args, _ in console.output if args and isinstance(args[0], str)
    ]
    assert any("503" in msg for msg in messages)


def test_connection_error(monkeypatch):
    console = FakeConsole()

    answers = iter(["question", "exit"])

    monkeypatch.setattr("src.api.client.Console", lambda: console)
    monkeypatch.setattr(
        "src.api.client.Prompt.ask", lambda *args, **kwargs: next(answers)
    )

    def fake_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError

    monkeypatch.setattr("src.api.client.requests.post", fake_post)
    monkeypatch.setattr(
        "src.api.client.sys.exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )
    monkeypatch.setattr("src.api.client.Status", DummyStatus)

    with pytest.raises(SystemExit):
        main()

    messages = [
        args[0] for args, _ in console.output if args and isinstance(args[0], str)
    ]
    assert any("Could not connect" in msg for msg in messages)


def test_keyboard_interrupt(monkeypatch):
    console = FakeConsole()

    monkeypatch.setattr("src.api.client.Console", lambda: console)

    def fake_prompt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("src.api.client.Prompt.ask", fake_prompt)
    monkeypatch.setattr(
        "src.api.client.sys.exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0


def test_environment_variables(monkeypatch):
    console = FakeConsole()

    monkeypatch.setenv("API_HOST", "localhost")
    monkeypatch.setenv("API_PORT", "9999")

    called = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "status": 200,
                "generated_response": '{"thought_process" : "thought" , "answer" : "hello" , "sources" : ["source"]}',
            }

    def fake_post(url, *args, **kwargs):
        called["url"] = url
        return FakeResponse()

    answers = iter(["question", "exit"])

    monkeypatch.setattr("src.api.client.Console", lambda: console)
    monkeypatch.setattr(
        "src.api.client.Prompt.ask", lambda *args, **kwargs: next(answers)
    )
    monkeypatch.setattr("src.api.client.requests.post", fake_post)
    monkeypatch.setattr(
        "src.api.client.sys.exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )
    monkeypatch.setattr("src.api.client.Status", DummyStatus)

    with pytest.raises(SystemExit):
        main()

    assert called["url"] == "http://localhost:9999/generate"
