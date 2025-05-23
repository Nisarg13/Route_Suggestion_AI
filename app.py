from __future__ import annotations
import os
from typing import List, Dict, Set, Tuple, Iterable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# ✅ Use static OpenAI API key (for demo)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Type alias for route strings or method-path tuples
Route = Union[str, Tuple[str, str]]

# General prompt for LLM to reason about routes
PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an intelligent assistant managing API Gateway routes.

Here are the existing routes:
{context}

The user asked:
"{question}"

Your job:
- If it's about route count → return the total number of unique routes.
- If it's about route existence → clearly confirm whether the route is in the list.
- If it's about suggesting a new route → return a RESTful path that avoids conflicts and minimizes resource usage.
- If it's not related → say so politely.

Always explain your reasoning clearly.
"""
)

@dataclass
class RouteManager:
    routes: List[Route]
    llm_model: str = "gpt-3.5-turbo"
    embed_model: str = "text-embedding-3-small"

    def answer(self, query: str) -> str:
        # Always use full context instead of RAG to avoid missing routes
        context_str = "\n".join(self._format_routes())
        prompt = PROMPT.format(question=query, context=context_str)
        llm = ChatOpenAI(model=self.llm_model)
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def conflicts(self) -> List[Tuple[str, str]]:
        groups = self._group_siblings()
        return self._find_conflicts(groups)

    def resources(self) -> Set[str]:
        resources = set("/")
        for route in self.routes:
            path = route[1] if isinstance(route, tuple) else route
            segments = path.strip("/").split("/")
            current = ""
            for segment in segments:
                current += f"/{segment}"
                resources.add(current)
        return resources

    def _format_routes(self) -> List[str]:
        return [r[1] if isinstance(r, tuple) else r for r in self.routes]

    def _group_siblings(self) -> Dict[str, Set[str]]:
        siblings = defaultdict(set)
        for route in self.routes:
            path = route[1] if isinstance(route, tuple) else route
            parent = "/".join(path.strip("/").split("/")[:-1]) or "/"
            siblings[parent].add(path)
        return {p: c for p, c in siblings.items() if len(c) > 1}

    def _find_conflicts(self, groups: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
        conflicts = []
        for children in groups.values():
            dynamic, static, proxy = set(), set(), set()
            for child in children:
                segment = child.split("/")[-1]
                if segment.startswith("{") and segment.endswith("+}"):
                    proxy.add(child)
                elif segment.startswith("{") and segment.endswith("}"):
                    dynamic.add(child)
                else:
                    static.add(child)
            conflicts += [(p, c) for p in proxy for c in children if p != c]
            conflicts += [(d, s) for d in dynamic for s in static]
        return conflicts

# ───────────── Streamlit UI ─────────────
st.set_page_config("API Route Assistant", "🛣️", layout="wide")
st.title("🛣️ API Route Management Assistant")

default_routes = """
/students
/students/{student_id}/accommodation
/students/{student_id}/notes
/teachers
/teachers/{teacher_id}/students
/teachers/{teacher_id}/notes
""".strip()

with st.expander("⚙️ Configure API Routes", expanded=True):
    route_text = st.text_area("Enter API routes (one per line):", default_routes, height=150)

routes_list = [line.strip() for line in route_text.splitlines() if line.strip()]

if "rm" not in st.session_state or st.session_state.get("cache") != routes_list:
    st.session_state["rm"] = RouteManager(routes_list)
    st.session_state["cache"] = routes_list

rm: RouteManager = st.session_state["rm"]

with st.form("query_form"):
    query = st.text_input("🔍 Ask a question about your API routes:")
    submitted = st.form_submit_button("Submit")

tab_response, tab_conflicts, tab_resources = st.tabs(["💬 Response", "⚠️ Conflicts", "📊 Resources"])

if submitted and query.strip():
    with tab_response:
        with st.spinner("Answering..."):
            st.markdown(rm.answer(query))

with tab_conflicts:
    conflicts = rm.conflicts()
    if conflicts:
        st.warning(f"{len(conflicts)} potential sibling conflicts found:")
        for a, b in conflicts:
            st.markdown(f"- `{a}` ↔ `{b}`")
    else:
        st.success("No sibling route conflicts found.")

with tab_resources:
    res = rm.resources()
    st.write(f"### Total API Gateway Resources: **{len(res)}**")
    LIMIT = 300
    if len(res) > LIMIT:
        st.error("⚠️ Exceeds API Gateway resource limit!")
    elif len(res) > LIMIT - 20:
        st.warning("⚠️ Nearing API Gateway limit.")
    else:
        st.success("✅ Resource usage is within limits.")
    with st.expander("🔍 View All Resources"):
        for r in sorted(res):
            st.markdown(f"- `{r}`")
