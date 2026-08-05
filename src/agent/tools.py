import ast
import logging
import operator
from abc import ABC, abstractmethod

from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, tool_input: str) -> str: ...


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.And: operator.and_,
    ast.Or: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_COMPARE_OPS = {
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}


def _safe_eval(node: ast.AST) -> int | float | bool:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Compare):
        current_left = _safe_eval(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _COMPARE_OPS:
                raise ValueError(
                    f"Unsupported comparison operator: {type(op).__name__}"
                )
            current_right = _safe_eval(comparator)
            if not _COMPARE_OPS[type(op)](current_left, current_right):
                return False
            current_left = current_right
        return True
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluates a basic arithmetic or logical expression using +, -, *, /, //, "
        "%, **, and, or, ^, <, >, <=, >=, ==, !=, is, is not, <<, >>, parentheses, "
        "and numbers or booleans. Example input: '(3 + 4) >= (2 << 1)'."
    )

    def run(self, tool_input: str) -> str:
        try:
            tree = ast.parse(tool_input, mode="eval")
        except SyntaxError as e:
            return (
                f"Error: '{tool_input}' is not a valid expression "
                f"({e.msg}). This tool only accepts numbers, booleans, "
                "and the operators + - * / // % ** and or ^ < > <= >= "
                "== != is 'is not' << >> with parentheses. Remove any "
                "units (e.g. 'km', '$'), commas, or other non-numeric "
                "text and pass only the numeric expression itself."
            )
        except Exception as e:  # noqa: BLE001
            return f"Error parsing through expression: {e}"
        try:
            result = _safe_eval(tree.body)
        except ZeroDivisionError:
            return "Error: division by zero."
        except ValueError as e:
            return (
                f"Error: {e}. This tool only accepts numbers, booleans, "
                "and the supported arithmetic/logical operators. "
                "Variable names, function calls, and other Python "
                "constructs are not allowed."
            )
        except Exception as e:  # noqa: BLE001
            return f"Error evaluating expression: {e}"
        return str(result)


class RetrievalTool(Tool):
    name = "retrieve_documents"
    description = (
        "Searches the knowledge base for relevant document chunks given "
        "a natural-language query. Use this when you need information "
        "that isn't already in the conversation."
    )

    def __init__(
        self, retriever: Retriever, reranker: Reranker, final_chunk_count: int
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.final_chunk_count = final_chunk_count

    def run(self, tool_input: str) -> str:
        if not tool_input.strip():
            return "Error: query must not be empty."
        try:
            retrieved = self.retriever.retrive_documents(tool_input)
            pairs = [(tool_input, doc) for doc in retrieved]
            reranked = self.reranker.rerank_documents(pairs, self.final_chunk_count)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Got Exception {e} in RetrievalTool.run")
            return f"Error retrieving documents: {e}"

        if not reranked:
            return "No relevant documents found."

        return "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
            for doc in reranked
        )


class FileLookupTool(Tool):
    name = "file_lookup"
    description = (
        "Looks up a specific ingested document by its exact source path "
        "and returns its full content. Use this after retrieve_documents "
        "has told you which source path you want to read in full to obtain greater context."
    )

    def __init__(self, documents_by_source: dict[str, str]) -> None:
        self.documents_by_source = documents_by_source

    def run(self, tool_input: str) -> str:
        content = self.documents_by_source.get(tool_input.strip())
        if content is None:
            return f"Error: no ingested document found with source '{tool_input}'."
        return content
