"""
Generate professional diagrams for the RAG Customer Support Assistant documentation.
Creates PNG images for: System Architecture, LangGraph Workflow, Data Flow, Decision Tree, HITL Flow.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color Palette ──
C = {
    "bg": "#FAFBFC",
    "ui": "#667EEA",
    "workflow": "#764BA2",
    "ai": "#F6993F",
    "data": "#38B2AC",
    "hitl": "#E53E3E",
    "text": "#1A202C",
    "white": "#FFFFFF",
    "light": "#EDF2F7",
    "success": "#38A169",
    "arrow": "#4A5568",
}


def _box(ax, x, y, w, h, label, color, fontsize=9, bold=False):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.15", facecolor=color,
                         edgecolor="#2D3748", linewidth=1.5, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=C["white"] if color not in [C["light"], C["bg"], "#FFF5F5", "#FFFAF0"] else C["text"],
            fontweight=weight, zorder=4, wrap=True,
            bbox=dict(boxstyle="round,pad=0", facecolor="none", edgecolor="none"))


def _arrow(ax, x1, y1, x2, y2, label="", color=None):
    """Draw an arrow between two points with optional label."""
    color = color or C["arrow"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=7, color=color,
                ha="left", va="center", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc=C["bg"], ec="none", alpha=0.9))


# ═══════════════════════════════════════════
# 1. System Architecture Diagram
# ═══════════════════════════════════════════
def gen_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    # Layer backgrounds
    layers = [
        (1, 7.2, 10, 1.3, "#EBF4FF", "USER INTERFACE LAYER"),
        (1, 5.4, 10, 1.3, "#F3E8FF", "WORKFLOW ORCHESTRATION (LangGraph)"),
        (1, 3.6, 10, 1.3, "#FFFAF0", "AI PROCESSING LAYER"),
        (1, 1.8, 10, 1.3, "#F0FFF4", "DATA STORAGE LAYER"),
        (1, 0.2, 10, 1.1, "#FFF5F5", "HITL ESCALATION LAYER"),
    ]
    for lx, ly, lw, lh, lc, lt in layers:
        rect = FancyBboxPatch((lx, ly), lw, lh, boxstyle="round,pad=0.1",
                              facecolor=lc, edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
        ax.add_patch(rect)
        ax.text(lx + 0.3, ly + lh - 0.15, lt, fontsize=8, fontweight="bold",
                color="#4A5568", va="top", zorder=2)

    # Layer 1: UI
    _box(ax, 3.5, 7.65, 2.2, 0.6, "Streamlit\nWeb UI", C["ui"], 9, True)
    _box(ax, 7, 7.65, 2.2, 0.6, "CLI\nInterface", C["ui"], 9, True)

    # Layer 2: Workflow
    _box(ax, 2.5, 5.85, 1.6, 0.55, "Classify\nIntent", C["workflow"], 8, True)
    _box(ax, 4.5, 5.85, 1.6, 0.55, "Retrieve\nDocuments", C["workflow"], 8, True)
    _box(ax, 6.5, 5.85, 1.6, 0.55, "Evaluate\nConfidence", C["workflow"], 8, True)
    _box(ax, 8.5, 5.85, 1.6, 0.55, "Generate\nResponse", C["workflow"], 8, True)

    # Layer 3: AI
    _box(ax, 4, 4.05, 2.5, 0.6, "Groq\nLlama 3.3 70B", C["ai"], 9, True)
    _box(ax, 8, 4.05, 2.8, 0.6, "HuggingFace\nall-MiniLM-L6-v2", C["ai"], 9, True)

    # Layer 4: Data
    _box(ax, 4, 2.25, 2.5, 0.6, "ChromaDB\nVector Store", C["data"], 9, True)
    _box(ax, 8, 2.25, 2.5, 0.6, "PDF Knowledge\nBase", C["data"], 9, True)

    # Layer 5: HITL
    _box(ax, 6, 0.55, 3.5, 0.55, "Escalation Engine\nTicket Creation → Human Response", C["hitl"], 8, True)

    # Arrows between layers
    _arrow(ax, 5, 7.3, 5, 6.15)
    _arrow(ax, 3.3, 5.55, 3.3, 4.4)
    _arrow(ax, 5.5, 5.55, 5.5, 4.4)
    _arrow(ax, 7.5, 5.55, 7.5, 4.4)
    _arrow(ax, 4, 3.7, 4, 2.6)
    _arrow(ax, 8, 3.7, 8, 2.6)

    # Arrows between workflow nodes
    _arrow(ax, 3.3, 5.85, 3.7, 5.85)
    _arrow(ax, 5.3, 5.85, 5.7, 5.85)
    _arrow(ax, 7.3, 5.85, 7.7, 5.85)

    # Escalation arrow
    _arrow(ax, 2.5, 5.55, 6, 0.85, "Escalate", C["hitl"])

    ax.set_title("System Architecture — RAG Customer Support Assistant",
                 fontsize=14, fontweight="bold", color=C["text"], pad=15)
    fig.savefig(os.path.join(OUT_DIR, "system_architecture.png"), dpi=180,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("✅ system_architecture.png")


# ═══════════════════════════════════════════
# 2. LangGraph Workflow Diagram
# ═══════════════════════════════════════════
def gen_workflow():
    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    # START
    circle = plt.Circle((1.5, 3.5), 0.4, color=C["success"], zorder=3)
    ax.add_patch(circle)
    ax.text(1.5, 3.5, "START", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold", zorder=4)

    # Nodes
    _box(ax, 3.5, 3.5, 1.8, 0.9, "1. Classify\nIntent", C["workflow"], 10, True)
    _box(ax, 6, 5, 1.8, 0.9, "2. Retrieve\nDocuments", C["data"], 10, True)
    _box(ax, 8.5, 5, 1.8, 0.9, "3. Evaluate\nConfidence", C["ai"], 10, True)
    _box(ax, 11, 5, 1.8, 0.9, "4. Generate\nResponse", C["success"], 10, True)
    _box(ax, 8.5, 2, 2.2, 0.9, "5. Escalate\nto Human (HITL)", C["hitl"], 10, True)

    # END nodes
    circle2 = plt.Circle((11, 3.5), 0.35, color="#2D3748", zorder=3)
    ax.add_patch(circle2)
    ax.text(11, 3.5, "END", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold", zorder=4)
    circle3 = plt.Circle((11, 1.2), 0.35, color="#2D3748", zorder=3)
    ax.add_patch(circle3)
    ax.text(11, 1.2, "END", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold", zorder=4)

    # Arrows
    _arrow(ax, 1.9, 3.5, 2.6, 3.5)
    _arrow(ax, 4.4, 3.95, 5.1, 4.75, "Normal Flow", C["success"])
    _arrow(ax, 4.4, 3.1, 7.4, 2.2, "COMPLAINT /\nESCALATE /\nOUT_OF_SCOPE", C["hitl"])
    _arrow(ax, 6.9, 5, 7.6, 5)
    _arrow(ax, 9.4, 5, 10.1, 5, "High\nConfidence", C["success"])
    _arrow(ax, 8.5, 4.5, 8.5, 2.5, "Low\nConfidence", C["hitl"])
    _arrow(ax, 11, 4.5, 11, 3.9)
    _arrow(ax, 9.6, 1.7, 10.65, 1.3)

    # Legend
    legend_items = [
        mpatches.Patch(color=C["workflow"], label="Intent Classification"),
        mpatches.Patch(color=C["data"], label="Document Retrieval"),
        mpatches.Patch(color=C["ai"], label="Confidence Evaluation"),
        mpatches.Patch(color=C["success"], label="Response Generation"),
        mpatches.Patch(color=C["hitl"], label="HITL Escalation"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8,
              framealpha=0.9, edgecolor="#CBD5E0")

    ax.set_title("LangGraph Workflow — StateGraph with Conditional Routing",
                 fontsize=14, fontweight="bold", color=C["text"], pad=15)
    fig.savefig(os.path.join(OUT_DIR, "langgraph_workflow.png"), dpi=180,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("✅ langgraph_workflow.png")


# ═══════════════════════════════════════════
# 3. Data Flow / RAG Pipeline Diagram
# ═══════════════════════════════════════════
def gen_dataflow():
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    # Ingestion pipeline (top row)
    ax.text(7, 4.6, "Document Ingestion Pipeline", ha="center", fontsize=11,
            fontweight="bold", color=C["text"])
    positions_top = [
        (1.5, 3.8, "PDF\nDocument", C["ui"]),
        (4, 3.8, "PyPDFLoader\n(Page Extract)", C["workflow"]),
        (6.5, 3.8, "Text Splitter\n(500 char chunks)", C["workflow"]),
        (9, 3.8, "HuggingFace\nEmbeddings", C["ai"]),
        (11.8, 3.8, "ChromaDB\n(384-dim vectors)", C["data"]),
    ]
    for x, y, label, color in positions_top:
        _box(ax, x, y, 2, 0.7, label, color, 8, True)

    for i in range(len(positions_top) - 1):
        x1 = positions_top[i][0] + 1
        x2 = positions_top[i+1][0] - 1
        _arrow(ax, x1, 3.8, x2, 3.8)

    # Query pipeline (bottom row)
    ax.text(7, 2.6, "Query Processing Pipeline", ha="center", fontsize=11,
            fontweight="bold", color=C["text"])
    positions_bot = [
        (1.5, 1.5, "User\nQuery", C["ui"]),
        (4, 1.5, "Intent\nClassifier", C["workflow"]),
        (6.5, 1.5, "ChromaDB\nRetrieval", C["data"]),
        (9, 1.5, "Confidence\nEvaluator", C["ai"]),
        (11.8, 1.5, "LLM Response\n(Groq Llama 3.3)", C["success"]),
    ]
    for x, y, label, color in positions_bot:
        _box(ax, x, y, 2, 0.7, label, color, 8, True)

    for i in range(len(positions_bot) - 1):
        x1 = positions_bot[i][0] + 1
        x2 = positions_bot[i+1][0] - 1
        _arrow(ax, x1, 1.5, x2, 1.5)

    # HITL branch
    _box(ax, 9, 0.3, 2, 0.5, "HITL Escalation", C["hitl"], 8, True)
    _arrow(ax, 9, 1.1, 9, 0.6, "Low", C["hitl"])

    ax.set_title("RAG Pipeline — Data Flow from PDF to Response",
                 fontsize=14, fontweight="bold", color=C["text"], pad=15)
    fig.savefig(os.path.join(OUT_DIR, "data_flow.png"), dpi=180,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("✅ data_flow.png")


# ═══════════════════════════════════════════
# 4. Decision Tree / Conditional Routing
# ═══════════════════════════════════════════
def gen_decision_tree():
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    # Decision diamonds
    def diamond(x, y, label, color):
        pts = [(x, y+0.5), (x+1.2, y), (x, y-0.5), (x-1.2, y)]
        poly = plt.Polygon(pts, facecolor=color, edgecolor="#2D3748", lw=1.5, zorder=3)
        ax.add_patch(poly)
        ax.text(x, y, label, ha="center", va="center", fontsize=7,
                color=C["white"], fontweight="bold", zorder=4)

    # Start
    _box(ax, 5.5, 7.2, 2, 0.6, "User Query\nReceived", C["ui"], 9, True)

    # Decision 1
    diamond(5.5, 5.8, "Intent =\nCOMPLAINT /\nESCALATE /\nOUT_OF_SCOPE?", C["workflow"])
    _arrow(ax, 5.5, 6.9, 5.5, 6.3)

    # YES → Escalate
    _box(ax, 9, 5.8, 2, 0.5, "ESCALATE\n(skip retrieval)", C["hitl"], 8, True)
    _arrow(ax, 6.7, 5.8, 8, 5.8, "YES", C["hitl"])

    # NO → Retrieve
    _box(ax, 5.5, 4.3, 2, 0.5, "Retrieve\nDocuments", C["data"], 8, True)
    _arrow(ax, 5.5, 5.3, 5.5, 4.6, "NO", C["success"])

    # Decision 2
    diamond(5.5, 3, "Documents\nFound?", C["workflow"])
    _arrow(ax, 5.5, 4.05, 5.5, 3.5)

    # NO → Escalate
    _box(ax, 9, 3, 2, 0.5, "ESCALATE\n(no context)", C["hitl"], 8, True)
    _arrow(ax, 6.7, 3, 8, 3, "NO", C["hitl"])

    # YES → Evaluate
    diamond(5.5, 1.5, "Top Score\n≥ 0.3?", C["workflow"])
    _arrow(ax, 5.5, 2.5, 5.5, 2, "YES", C["success"])

    # NO → Escalate
    _box(ax, 9, 1.5, 2, 0.5, "ESCALATE\n(low confidence)", C["hitl"], 8, True)
    _arrow(ax, 6.7, 1.5, 8, 1.5, "NO", C["hitl"])

    # YES → Generate
    _box(ax, 5.5, 0.3, 2.2, 0.5, "GENERATE\nRESPONSE ✅", C["success"], 9, True)
    _arrow(ax, 5.5, 1, 5.5, 0.6, "YES", C["success"])

    ax.set_title("Conditional Routing — Decision Tree",
                 fontsize=14, fontweight="bold", color=C["text"], pad=15)
    fig.savefig(os.path.join(OUT_DIR, "decision_tree.png"), dpi=180,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("✅ decision_tree.png")


# ═══════════════════════════════════════════
# 5. HITL Escalation Flow
# ═══════════════════════════════════════════
def gen_hitl_flow():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    steps = [
        (1.5, 4.5, "Trigger\nDetected", C["hitl"]),
        (4, 4.5, "Create\nEscalation\nTicket", C["workflow"]),
        (6.5, 4.5, "Notify\nCustomer", C["ui"]),
        (9, 4.5, "Show HITL\nPanel", C["ai"]),
        (9, 2.5, "Human Agent\nTypes Response", C["success"]),
        (6.5, 2.5, "Integrate\nResponse", C["workflow"]),
        (4, 2.5, "Update Chat\nHistory", C["data"]),
        (1.5, 2.5, "Resume\nNormal Flow", C["success"]),
    ]

    for x, y, label, color in steps:
        _box(ax, x, y, 1.8, 0.8, label, color, 8, True)

    # Arrows (top row L→R)
    for i in range(3):
        _arrow(ax, steps[i][0]+0.9, steps[i][1], steps[i+1][0]-0.9, steps[i+1][1])
    # Down
    _arrow(ax, 9, 4.05, 9, 2.95)
    # Bottom row R→L
    for i in range(4, 7):
        _arrow(ax, steps[i][0]-0.9, steps[i][1], steps[i+1][0]+0.9, steps[i+1][1])

    # Trigger labels
    triggers = [
        "• COMPLAINT intent",
        "• ESCALATE intent",
        "• OUT_OF_SCOPE intent",
        "• Low retrieval confidence (<0.3)",
        "• No documents found",
        "• LLM generation failure",
    ]
    ax.text(1.5, 1.2, "Escalation Triggers:", fontsize=9, fontweight="bold",
            color=C["text"], va="top")
    for i, t in enumerate(triggers):
        ax.text(1.5, 0.9 - i*0.2, t, fontsize=7, color=C["text"], va="top")

    # Ticket contents
    ax.text(6.5, 1.2, "Ticket Contains:", fontsize=9, fontweight="bold",
            color=C["text"], va="top")
    ticket_items = ["• ticket_id, timestamp", "• customer_query, intent",
                    "• retrieval_confidence", "• escalation_reason", "• status: PENDING"]
    for i, t in enumerate(ticket_items):
        ax.text(6.5, 0.9 - i*0.2, t, fontsize=7, color=C["text"], va="top")

    ax.set_title("HITL Escalation Flow — Human-in-the-Loop Process",
                 fontsize=14, fontweight="bold", color=C["text"], pad=15)
    fig.savefig(os.path.join(OUT_DIR, "hitl_flow.png"), dpi=180,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("✅ hitl_flow.png")


if __name__ == "__main__":
    print("Generating diagrams...")
    gen_architecture()
    gen_workflow()
    gen_dataflow()
    gen_decision_tree()
    gen_hitl_flow()
    print(f"\n✅ All diagrams saved to: {OUT_DIR}")
