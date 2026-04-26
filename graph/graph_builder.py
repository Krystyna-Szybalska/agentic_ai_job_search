from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import JobSearchState
from graph.nodes import (
    parse_cv,
    embed_cv,
    retrieve_jobs,
    matching_agent,
    critic_agent,
    generate_report_node,
    fetch_remote_jobs,
)
from config.settings import JOB_SOURCE


def build_graph():
    builder = StateGraph(JobSearchState)

    builder.add_node("parse_cv", parse_cv)
    builder.add_node("matching_agent", matching_agent)
    builder.add_node("critic_agent", critic_agent)
    builder.add_node("generate_report", generate_report_node)

    if JOB_SOURCE == "remotive":
        builder.add_node("fetch_remote_jobs", fetch_remote_jobs)
        builder.add_edge(START, "parse_cv")
        builder.add_edge("parse_cv", "fetch_remote_jobs")
        builder.add_edge("fetch_remote_jobs", "matching_agent")
    else:
        builder.add_node("embed_cv", embed_cv)
        builder.add_node("retrieve_jobs", retrieve_jobs)
        builder.add_edge(START, "parse_cv")
        builder.add_edge("parse_cv", "embed_cv")
        builder.add_edge("embed_cv", "retrieve_jobs")
        builder.add_edge("retrieve_jobs", "matching_agent")

    builder.add_edge("matching_agent", "critic_agent")
    builder.add_edge("critic_agent", "generate_report")
    builder.add_edge("generate_report", END)

    interrupts = ["generate_report"]
    if JOB_SOURCE == "remotive":
        interrupts.insert(0, "fetch_remote_jobs")

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupts,
    )
    return graph
