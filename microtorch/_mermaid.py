class Mermaid:
    def __init__(self, diagram: str, name: str):
        self._diagram = self._process_diagram(diagram)
        self._name = name

    @staticmethod
    def _process_diagram(diagram: str) -> str:
        _diagram = diagram.replace("\n", "\\n")
        _diagram = _diagram.lstrip("\\n")
        _diagram = _diagram.replace("'", '"')
        return _diagram

    def __repr__(self) -> str:
        ret = f"""
        <div class="mermaid-{self._name}" style="text-align: center;"></div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.1.0/+esm'
            const graphDefinition = \'_diagram\';
            const element = document.querySelector('.mermaid-{self._name}');
            const {{ svg }} = await mermaid.render('graphDiv-{self._name}', graphDefinition);
            element.innerHTML = svg;
        </script>
        """
        ret = ret.replace("_diagram", self._diagram)
        return ret
