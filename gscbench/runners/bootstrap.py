"""Framework bootstrap helpers."""

from importlib import import_module

from gscbench.models.registry import MODEL_REGISTRY, register_model


MODEL_SPECS = {
    "hungarian": ("gscbench.solvers.ged.hungarian", "HungarianGEDModel", "HungarianGEDConfig", "HungarianGEDDataAdapter"),
    "vj": ("gscbench.solvers.ged.vj", "VJGEDModel", "VJGEDConfig", "VJGEDDataAdapter"),
    "beam": ("gscbench.solvers.ged.beam", "BeamSearchGEDModel", "BeamSearchGEDConfig", "BeamSearchGEDDataAdapter"),
    "fgwalign": ("gscbench.solvers.ged.fgwalign", "FGWAlignModel", "FGWAlignConfig", "FGWAlignDataAdapter"),
    "astar": ("gscbench.solvers.ged.astar", "Astar", "AstarConfig", "AstarDataAdapter"),
    "eric": ("gscbench.models.eric", "ERICModel", None, "ERICDataAdapter"),
    "egsc": ("gscbench.models.egsc", "EGSCModel", None, "EGSCDataAdapter"),
    "gedgnn": ("gscbench.models.gedgnn", "GEDGNNModel", None, "GEDGNNDataAdapter"),
    "gedranker": ("gscbench.models.gedranker", "GEDRankerModel", None, "GEDRankerDataAdapter"),
    "gelato": ("gscbench.models.gelato", "GelatoModel", None, "GelatoDataAdapter"),
    "gen": ("gscbench.models.gen", "GENModel", None, "GENDataAdapter"),
    "genn_astar": ("gscbench.models.genn_astar", "GENNAStarModel", None, "GENNAStarDataAdapter"),
    "gediot": ("gscbench.models.gediot", "GEDIOTModel", None, "GEDIOTDataAdapter"),
    "gmn": ("gscbench.models.gmn", "GMNModel", None, "GMNDataAdapter"),
    "gotsim": ("gscbench.models.gotsim", "GOTSimModel", None, "GOTSimDataAdapter"),
    "graphedx": ("gscbench.models.graphedx", "GraphEdXModel", None, "GraphEdXDataAdapter"),
    "graph2region": ("gscbench.models.graph2region", "Graph2RegionModel", None, "Graph2RegionDataAdapter"),
    "grasp": ("gscbench.models.grasp", "GraSPModel", None, "GraSPDataAdapter"),
    "greed": ("gscbench.models.greed", "GreedModel", None, "GreedDataAdapter"),
    "h2mn": ("gscbench.models.h2mn", "H2MNModel", None, "H2MNDataAdapter"),
    "mata": ("gscbench.models.mata", "MATAModel", None, "MATADataAdapter"),
    "noah": ("gscbench.models.noah", "NoahModel", None, "NoahDataAdapter"),
    "graphsim": ("gscbench.models.graphsim", "GraphSimModel", None, "GraphSimDataAdapter"),
    "simgnn": ("gscbench.models.simgnn", "SimGNNModel", None, "SimGNNDataAdapter"),
    "tagsim": ("gscbench.models.tagsim", "TaGSimModel", None, "TaGSimDataAdapter"),
}


def bootstrap(*names: str) -> None:
    for name in names or MODEL_SPECS:
        if name in MODEL_REGISTRY:
            continue
        module_name, model_name, config_name, adapter_name = MODEL_SPECS[name]
        module = import_module(module_name)
        register_model(
            name, getattr(module, model_name),
            config_cls=getattr(module, config_name) if config_name else None,
            adapter_cls=getattr(module, adapter_name),
        )
