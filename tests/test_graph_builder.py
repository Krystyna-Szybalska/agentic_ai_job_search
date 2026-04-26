from unittest.mock import patch


class TestGraphRouting:
    @patch("graph.graph_builder.JOB_SOURCE", "local")
    def test_local_source_includes_embed_and_retrieve(self):
        from graph.graph_builder import build_graph
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "embed_cv" in node_names
        assert "retrieve_jobs" in node_names

    @patch("graph.graph_builder.JOB_SOURCE", "remotive")
    def test_remotive_source_includes_fetch_remote(self):
        from graph.graph_builder import build_graph
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "fetch_remote_jobs" in node_names


class TestInterrupts:
    @patch("graph.graph_builder.JOB_SOURCE", "remotive")
    def test_remotive_source_interrupts_before_fetch_and_report(self):
        from graph.graph_builder import build_graph
        graph = build_graph()
        interrupts = list(graph.interrupt_before_nodes)
        assert "fetch_remote_jobs" in interrupts
        assert "generate_report" in interrupts

    @patch("graph.graph_builder.JOB_SOURCE", "local")
    def test_local_source_only_interrupts_before_report(self):
        from graph.graph_builder import build_graph
        graph = build_graph()
        interrupts = list(graph.interrupt_before_nodes)
        assert "fetch_remote_jobs" not in interrupts
        assert "generate_report" in interrupts
