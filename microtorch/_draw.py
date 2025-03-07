from typing import Literal
from ._autograd import Tensor
from ._mermaid import Mermaid

_node_colors = {
    "const": ("#E3F2FD", "#0D47A1"),
    "leaf": ("#B3E5FC", "#00796B"),
    "op": ("#ECEFF1", "#546E7A"),
}


def get_node_info(node: Tensor) -> tuple[str, str, str | None]:
    node_name = node.name
    node_data = f"{node.data.shape}"
    node_args = None
    if node.op is not None and len(node.op.op_args) != 0:
        op_args = [f"{k}={v}" for k, v in node.op.op_args.items()]
        node_args = ", ".join(op_args)
    return node_name, node_args, node_data


def get_mermaid_node_info(node: Tensor) -> str:
    node_id = str(id(node))
    node_name, node_args, node_data = get_node_info(node)
    if node_args is None:
        label = f"<b>{node_name}</b><br><small>{node_data}</small>"
    else:
        label = f"<b>{node_name}</b><br><small>{node_args}<br>{node_data}</small>"
    return f"{node_id}(\"{label}\")"


def get_mermaid_node_style(node: Tensor) -> str:
    node_id = str(id(node))
    if not node.requires_grad:
        fill_color, stroke_color = _node_colors["const"]
    elif node.op is None:
        fill_color, stroke_color = _node_colors["leaf"]
    else:
        fill_color, stroke_color = _node_colors["op"]
    return f"style {node_id} fill:{fill_color},stroke:{stroke_color}"


def _build_mermaid_script(node: Tensor, _script: str) -> str:
    if node.src is None:
        return ""

    for src_node in node.src:
        if src_node is None:
            continue
        node_info = get_mermaid_node_info(src_node)
        if node_info not in _script:
            _script += node_info + "\n"
            style = get_mermaid_node_style(src_node)
            _script += style + "\n"

        edge = f"{str(id(src_node))}-->{str(id(node))}\n"
        if edge not in _script:
            _script += edge

        if src_node.src is not None:
            _script = _build_mermaid_script(src_node, _script)

    return _script


def build_mermaid_script(root_node: Tensor, orientation: Literal["LR", "RL", "TB", "BT"] = "TB") -> str:
    _script = f"graph {orientation}\n"
    _script += get_mermaid_node_info(root_node) + "\n"
    _script += get_mermaid_node_style(root_node) + "\n"

    _script = _build_mermaid_script(root_node, _script)

    return _script


def draw_to_html(root_node: Tensor, name: str, orientation: Literal["LR", "RL", "TB", "BT"] = "TB") -> None:
    mermaid_script = build_mermaid_script(root_node, orientation=orientation)
    html = Mermaid(mermaid_script, name)
    with open(name + ".html", "w", encoding="utf-8") as f:
        f.write(repr(html))
