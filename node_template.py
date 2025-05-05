from streamlit_flow.elements import StreamlitFlowNode
import streamlit as st
import secrets

COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

NODE_TYPE_COLORS = {
    "theme": "#FF4D00",       # orange
    "question": "#4D96FF",    # blue
    "terminal": "#6BCB77"     # green
}
import numpy as np
from streamlit_flow.elements import StreamlitFlowNode
from utils import cosine_similarity

class BaseNode:
    def __init__(self, node_id, label, full_question, category, node_type, parent_id=None, color=None, metadata=None):
        self.node_id = node_id
        self.label = label
        self.full_question = full_question
        self.category = category
        self.node_type = node_type  # Must be: "theme", "question", or "terminal"
        self.parent_id = parent_id
        self.color = color or NODE_TYPE_COLORS.get(node_type, secrets.choice(COLOR_PALETTE))
        self.metadata = metadata or {}
        self.expanded = False

    def mark_expanded(self):
        self.expanded = True

    def can_expand(self):
        return not self.expanded and self.node_type != "terminal"

    def to_streamlit_node(self):
        return StreamlitFlowNode(
            self.node_id,
            (secrets.SystemRandom().randint(-100, 100), secrets.SystemRandom().randint(-100, 100)),
            {
                "section_path": self.node_id,
                "short_label": self.label,
                "full_question": self.full_question,
                "category": self.category,
                "node_type": self.node_type,
                "metadata": self.metadata,
                "content": f"**{self.label}**"
            },
            "default",
            "right",
            "left",
            style={"backgroundColor": self.color}
        )

    def get_children(self, *args, **kwargs):
        return []


class ThemeNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_children(self, openai_client, metadata_string):
        from question_generator_module import generate_theme_questions_and_labels

        questions, labels, node_types = generate_theme_questions_and_labels(
            self.full_question,
            metadata_string,
            client=openai_client
        )

        children = []
        for i, (q, lbl, ntype) in enumerate(zip(questions, labels, node_types), 1):

            embedding = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[q]
            ).data[0].embedding

            # ❌ Skip if too similar to a previously generated node
            is_duplicate = any(
                cosine_similarity(embedding, existing_vec) > 0.92
                for existing_vec, _, _ in st.session_state.seen_embeddings
            )
            if is_duplicate:
                continue

            node_id = f"{self.node_id}.{i}"

            if ntype == "theme":
                child = ThemeNode(
                    node_id=node_id,
                    label=lbl,
                    full_question=q,
                    category=self.category,
                    node_type="theme",
                    parent_id=self.node_id
                )
            elif ntype == "question":
                child = QuestionNode(
                    node_id=node_id,
                    label=lbl,
                    full_question=q,
                    category=self.category,
                    node_type="question",
                    parent_id=self.node_id
                )
            else:
                child = TerminalNode(
                    node_id=node_id,
                    label=lbl,
                    full_question=q,
                    category=self.category,
                    node_type="terminal",
                    parent_id=self.node_id
                )

            children.append(child)

        return children


class QuestionNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_children(self, openai_client, metadata_string):
        from question_generator_module import generate_followup_questions_and_labels

        current_depth = self.node_id.count(".")

        questions, labels, node_types = generate_followup_questions_and_labels(
            prompt=self.full_question,
            metadata_string=metadata_string,
            client=openai_client
        )

        children = []
        for i, (q, lbl, ntype) in enumerate(zip(questions, labels, node_types), 1):
            # 🔍 Compute embedding for the new question
            embedding = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[q]
            ).data[0].embedding

            # ❌ Skip if too similar to any previous question
            is_duplicate = any(
                cosine_similarity(embedding, existing_vec) > 0.92
                for existing_vec, _, _ in st.session_state.seen_embeddings
            )
            if is_duplicate:
                continue

            # ✅ Otherwise, register embedding + create node
            node_id = f"{self.node_id}.{i}"
            st.session_state.seen_embeddings.append((embedding, q, node_id))

            if ntype == "question" and current_depth < 2:
                child = QuestionNode(
                    node_id=node_id,
                    label=lbl,
                    full_question=q,
                    category=self.category,
                    node_type="question",
                    parent_id=self.node_id
                )
            else:
                child = TerminalNode(
                    node_id=node_id,
                    label=lbl,
                    full_question=q,
                    category=self.category,
                    node_type="terminal",
                    parent_id=self.node_id
                )

            children.append(child)

        return children

class TerminalNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def can_expand(self):
        return False

    def get_children(self, *args, **kwargs):
        return []
