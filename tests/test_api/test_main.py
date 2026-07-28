import asyncio

import src.api.main as main_module


def test_lifespan_runs_init_and_cleanup():
    async def run_lifespan():
        async with main_module.lifespan(main_module.app):
            return

    asyncio.run(run_lifespan())


def test_main_calls_uvicorn_run_with_configured_settings(monkeypatch):
    calls = {}

    def fake_run(app, host, port, log_level):
        calls["host"] = host
        calls["port"] = port
        calls["log_level"] = log_level

    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

    main_module.main()

    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 8000
    assert calls["log_level"] == "INFO"
