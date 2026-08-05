import pytest

from src.agent.tools import CalculatorTool, FileLookupTool, RetrievalTool

# ---------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 2", "4"),
        ("(3 + 4) * 2", "14"),
        ("10 / 4", "2.5"),
        ("2 ** 10", "1024"),
        ("7 % 3", "1"),
        ("7 // 2", "3"),
        ("-5 + 3", "-2"),
        ("-(2 + 3)", "-5"),
        ("(3 + 4) >= (2 << 1)", "True"),
        ("1 < 7 < 5", "False"),
    ],
)
def test_calculator_tool_valid_expressions(expression, expected):
    tool = CalculatorTool()

    assert tool.run(expression) == expected


def test_calculator_tool_division_by_zero():
    tool = CalculatorTool()

    result = tool.run("1 / 0")

    assert "division by zero" in result.lower()


def test_calculator_tool_non_string_input_raises():
    tool = CalculatorTool()

    result = tool.run(None)  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    assert result.startswith("Error")
    assert "parsing" in result


def test_calculator_tool_rejects_non_arithmetic_syntax():
    tool = CalculatorTool()

    result = tool.run("__import__('os').system('echo pwned')")

    assert result.startswith("Error")


def test_calculator_tool_embedded_units_gives_actionable_message():
    # The exact case observed in practice: the model included a unit
    # ("ms") directly in the numeric literal. The message must explain
    # *why* this failed and *what to do differently*, not just relay
    # Python's raw parser error -- this is what the agent's next step
    # actually reads and reacts to.
    tool = CalculatorTool()

    result = tool.run("12ms * 250")

    assert result.startswith("Error:")
    assert "not a valid expression" in result
    assert "units" in result.lower()
    assert "only the numeric expression" in result.lower()


def test_calculator_tool_currency_symbol_gives_actionable_message():
    tool = CalculatorTool()

    result = tool.run("$50 + 10")

    assert "not a valid expression" in result
    assert "units" in result.lower()


def test_calculator_tool_unbalanced_parens_gives_actionable_message():
    tool = CalculatorTool()

    result = tool.run("(3 + 4")

    assert "not a valid expression" in result


def test_calculator_tool_syntax_error_and_semantic_error_have_distinct_messages():
    tool = CalculatorTool()

    syntax_error_msg = tool.run("63km * 36")
    semantic_error_msg = tool.run("x + 1")

    assert "not a valid expression" in syntax_error_msg
    assert "not a valid expression" not in semantic_error_msg
    assert "Variable names, function calls" in semantic_error_msg


def test_calculator_tool_safe_eval_raises_exception(monkeypatch):
    tool = CalculatorTool()

    def dummy_eval(expression):
        raise RuntimeError("genexcept")

    monkeypatch.setattr("src.agent.tools._safe_eval", dummy_eval)
    out = tool.run("63 * 36")
    assert out == "Error evaluating expression: genexcept"


def test_calculator_tool_rejects_name_references():
    tool = CalculatorTool()

    result = tool.run("os.system('ls')")

    assert result.startswith("Error")


def test_calculator_tool_rejects_function_calls():
    tool = CalculatorTool()

    result = tool.run("print(1)")

    assert result.startswith("Error")


def test_calculator_tool_rejects_invalid_value():
    tool = CalculatorTool()

    result = tool.run("'Invalid'")

    assert result.startswith("Error")


def test_calculator_tool_rejects_unsupported_comparison():
    tool = CalculatorTool()

    result = tool.run("1 in 2")

    assert result.startswith("Error")


def test_calculator_tool_malformed_expression():
    tool = CalculatorTool()

    result = tool.run("2 + ")

    assert result.startswith("Error")


def test_calculator_tool_has_name_and_description():
    tool = CalculatorTool()

    assert tool.name == "calculator"
    assert len(tool.description) > 0


# ---------------------------------------------------------------------
# RetrievalTool
# ---------------------------------------------------------------------


class _FakeDoc:
    def __init__(self, content: str, source: str):
        self.page_content = content
        self.metadata = {"source": source}


def test_retrieval_tool_returns_formatted_chunks():
    class FakeRetriever:
        def retrive_documents(self, query):
            return [_FakeDoc("chunk one", "a.py")]

    class FakeReranker:
        def rerank_documents(self, pairs, n):
            return [doc for _, doc in pairs]

    tool = RetrievalTool(FakeRetriever(), FakeReranker(), final_chunk_count=3)  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    result = tool.run("what does X do")

    assert "[a.py]" in result
    assert "chunk one" in result


def test_retrieval_tool_no_results():
    class FakeRetriever:
        def retrive_documents(self, query):
            return []

    class FakeReranker:
        def rerank_documents(self, pairs, n):
            return []

    tool = RetrievalTool(FakeRetriever(), FakeReranker(), final_chunk_count=3)  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    result = tool.run("something obscure")

    assert result == "No relevant documents found."


def test_retrieval_tool_empty_query():
    tool = RetrievalTool(retriever=None, reranker=None, final_chunk_count=3)  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    result = tool.run("   ")

    assert result.startswith("Error")


def test_retrieval_tool_retriever_failure_returns_error_string():
    class FailingRetriever:
        def retrive_documents(self, query):
            raise RuntimeError("index unavailable")

    tool = RetrievalTool(FailingRetriever(), reranker=None, final_chunk_count=3)  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    result = tool.run("query")

    assert result.startswith("Error")
    assert "index unavailable" in result


# ---------------------------------------------------------------------
# FileLookupTool
# ---------------------------------------------------------------------


def test_file_lookup_tool_found():
    tool = FileLookupTool({"a.py": "print('hi')"})

    assert tool.run("a.py") == "print('hi')"


def test_file_lookup_tool_not_found():
    tool = FileLookupTool({"a.py": "print('hi')"})

    result = tool.run("missing.py")

    assert result.startswith("Error")


def test_file_lookup_tool_strips_whitespace():
    tool = FileLookupTool({"a.py": "content"})

    assert tool.run("  a.py  ") == "content"
