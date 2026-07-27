"""HPOA 导入的来源隔离测试。"""

from unittest.mock import Mock

from scripts.import_hpoa import _write_gene_rels, _write_hpoa_rels


def test_hpoa_relations_match_source_scoped_hpo_symptoms():
    graph = Mock()

    _write_hpoa_rels(graph, [{"database_id": "OMIM:1", "hpo_id": "HP:1"}])

    cypher = graph.query.call_args.args[0]
    assert "MMSymptom {source_db: 'HPO', hpo_id: r.hpo_id}" in cypher
    assert "SET rel.source_db = 'HPO'" in cypher


def test_gene_relations_match_source_scoped_hpo_nodes():
    graph = Mock()

    _write_gene_rels(
        graph,
        [{"ncbi_gene_id": "NCBI:1", "gene_symbol": "GENE", "hpo_id": "HP:1"}],
    )

    cypher = graph.query.call_args.args[0]
    assert "Target {source_db: 'HPO', ncbi_gene_id: r.ncbi_gene_id}" in cypher
    assert "MMSymptom {source_db: 'HPO', hpo_id: r.hpo_id}" in cypher
    assert "Target {ncbi_gene_id: r.ncbi_gene_id}" in cypher
    assert "bridged.source_db IN ['ITCM', 'SymMap']" in cypher
    assert "bridge.identity_bridge = 'ncbi_id'" in cypher
