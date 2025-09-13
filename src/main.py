from src.graph import stateGraph
from src.state import initialState

app = stateGraph.compile()
print(app.get_graph().draw_mermaid())

app.invoke(initialState)
