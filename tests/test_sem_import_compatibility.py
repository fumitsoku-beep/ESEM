from psysem import (
    SEMModel,
    SEMResult,
    build_measurement_design,
    build_structural_design,
    compute_basic_fit_indices,
    estimate_parameter_inference,
    parse_model,
    sem,
    to_markdown,
)
from psysem.sem.core import SEMModel as NewSEMModel, sem as new_sem
from psysem.sem.estimation import SEMFitConfig, optimize_ml_parameters
from psysem.sem.fit_indices import compute_basic_fit_indices as new_compute_basic_fit_indices
from psysem.sem.inference import estimate_parameter_inference as new_estimate_parameter_inference
from psysem.sem.measurement import build_measurement_design as new_build_measurement_design
from psysem.sem.model import ModelSpec, parse_model as new_parse_model
from psysem.sem.reporting import to_markdown as new_to_markdown
from psysem.sem.result import SEMResult as NewSEMResult
from psysem.sem.structural import build_structural_design as new_build_structural_design


def test_top_level_sem_api_still_importable() -> None:
    assert SEMModel is not None
    assert SEMResult is not None
    assert callable(sem)
    assert callable(to_markdown)
    assert callable(parse_model)
    assert callable(compute_basic_fit_indices)
    assert callable(build_measurement_design)
    assert callable(build_structural_design)
    assert callable(estimate_parameter_inference)


def test_top_level_sem_api_reexports_new_sem_implementations() -> None:
    assert SEMModel is NewSEMModel
    assert sem is new_sem
    assert SEMResult is NewSEMResult
    assert compute_basic_fit_indices is new_compute_basic_fit_indices
    assert estimate_parameter_inference is new_estimate_parameter_inference
    assert build_measurement_design is new_build_measurement_design
    assert build_structural_design is new_build_structural_design
    assert parse_model is new_parse_model
    assert to_markdown is new_to_markdown


def test_sem_subsystem_public_modules_are_importable() -> None:
    assert NewSEMModel is not None
    assert NewSEMResult is not None
    assert ModelSpec is not None
    assert callable(optimize_ml_parameters)
    assert callable(new_build_measurement_design)
    assert callable(new_build_structural_design)


def test_sem_subsystem_exports_stay_coherent() -> None:
    assert SEMFitConfig.__module__.startswith("psysem.sem.")
    assert NewSEMModel.__module__ == "psysem.sem.core"
    assert NewSEMResult.__module__ == "psysem.sem.result"
    assert new_build_measurement_design.__module__ == "psysem.sem.measurement.builder"
    assert new_build_structural_design.__module__ == "psysem.sem.structural.builder"
