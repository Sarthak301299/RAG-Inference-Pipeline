from src.generation import schemas


class Dummy:
    pass


def test_make_output_schema(monkeypatch):
    monkeypatch.setattr(schemas, "BaseModel", Dummy)

    out = schemas.make_output_schema(2000, 14000)

    assert hasattr(out, "thought_process")
    assert hasattr(out, "answer")
    assert hasattr(out, "sources")
