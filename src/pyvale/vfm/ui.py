from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from pyvale.vfm.mechanical_properties import (
    ConstituitiveLaw,
    ParameterName,
    required_parameters_for_law,
)
from pyvale.vfm.project_definition import (
    IdentificationProject,
    MetricSpec,
    OptimiserSpec,
    ParameterisationSpec,
    PhaseDefinition,
    create_default_parameter_definition,
    create_default_phase_definition,
)
from pyvale.vfm.project_io import load_project, project_to_yaml_text, save_project


PARAMETERISATION_KINDS = [
    "known",
    "homogeneous",
    "mesh",
    "basis_function",
    "slice_wise",
    "linked",
]

EXCLUSIVE_PARAMETERISATION_KINDS = {"known", "linked"}

METRIC_KINDS = [
    "sbvf",
    "egi",
    "fre",
    "udvf_uniform",
    "udvf_slicewise",
    "udvf_piecewise",
]

OPTIMISER_KINDS = ["least_squares", "pattern_search"]
INITIAL_VALUE_TYPES = ["float", "2d np array"]


@dataclass(frozen=True, slots=True)
class ParameterisationRowRef:
    parameter_name: str
    spec_index: int


@dataclass(frozen=True, slots=True)
class DetailField:
    key: str
    label: str
    value: str
    editor_kind: str = "text"
    choices: tuple[str, ...] = ()
    placeholder: str = ""


def _format_scalar(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:g}"


def _format_initial_value(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{value:g}"


def _parse_float(text: str, default: float | None = None) -> float | None:
    stripped = text.strip()
    if not stripped:
        return default
    try:
        return float(stripped)
    except ValueError:
        return default


def _parse_int(text: str, default: int) -> int:
    stripped = text.strip()
    if not stripped:
        return default
    try:
        return int(stripped)
    except ValueError:
        return default


def _parse_int_pair(text: str, default: list[int]) -> list[int]:
    cleaned = text.replace("x", ",").replace("X", ",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != 2:
        return default
    try:
        return [int(parts[0]), int(parts[1])]
    except ValueError:
        return default


def _parse_number_list(text: str, default: list[int]) -> list[int]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return default
    try:
        return [int(part) for part in parts]
    except ValueError:
        return default


def _parse_csv(text: str, default: list[str]) -> list[str]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return default
    return parts


def _readonly_item(text: str) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
    return item


def _required_parameter_names(constituitive_law: ConstituitiveLaw) -> list[str]:
    return [parameter_name.name for parameter_name in required_parameters_for_law(constituitive_law)]


def _default_parameterisation_spec(
    constituitive_law: ConstituitiveLaw,
    parameter_name: str,
) -> ParameterisationSpec:
    default_phase = create_default_phase_definition(constituitive_law)
    specs = default_phase.parameterisations.get(parameter_name)
    if specs:
        return deepcopy(specs[0])
    return ParameterisationSpec(kind="known")


def _stackable_parameterisation_spec(parameter_name: str) -> ParameterisationSpec:
    return ParameterisationSpec(
        kind="homogeneous",
        options={"initialise_from": "initial_value"},
        name=f"{parameter_name}_layer",
    )


def _parameterisation_spec_for_kind(
    parameter_name: str,
    kind: str,
) -> ParameterisationSpec:
    if kind == "known":
        return ParameterisationSpec(kind="known")
    if kind == "homogeneous":
        return ParameterisationSpec(
            kind="homogeneous",
            options={"initialise_from": "initial_value"},
        )
    if kind == "mesh":
        return ParameterisationSpec(
            kind="mesh",
            options={"initial_size": [2, 2], "element_order": 0},
        )
    if kind == "basis_function":
        return ParameterisationSpec(
            kind="basis_function",
            options={
                "kernel_shape": "univariate",
                "initial_count": 1,
                "addition_method": "place",
                "optimisation_strategy": "all_together",
            },
        )
    if kind == "slice_wise":
        return ParameterisationSpec(
            kind="slice_wise",
            options={"num_slices": 5, "direction": "x"},
        )
    if kind == "linked":
        return ParameterisationSpec(
            kind="linked",
            source_parameter=parameter_name,
            free_dof_groups=["value"],
        )
    return ParameterisationSpec(kind=kind)


def _parameterisation_rows(phase: PhaseDefinition) -> list[ParameterisationRowRef]:
    rows: list[ParameterisationRowRef] = []
    for parameter_name, specs in phase.parameterisations.items():
        for spec_index, _ in enumerate(specs):
            rows.append(ParameterisationRowRef(parameter_name, spec_index))
    return rows


def _metric_spec_for_kind(kind: str) -> MetricSpec:
    if kind == "sbvf":
        return MetricSpec(
            kind="sbvf",
            weight=1.0,
            options={
                "virtual_mesh_size": [15, 15],
                "stress_sensitivity": "total",
                "perturb_type": "dof",
                "perturbation_factor": 0.15,
            },
        )
    if kind == "egi":
        return MetricSpec(
            kind="egi",
            weight=1.0,
            options={"window_sizes": [5, 9, 13], "num_windows": 3},
        )
    if kind == "fre":
        return MetricSpec(
            kind="fre",
            weight=1.0,
            options={"points_per_slice": 20, "direction": "x"},
        )
    if kind == "udvf_uniform":
        return MetricSpec(kind="udvf_uniform", weight=1.0)
    if kind == "udvf_slicewise":
        return MetricSpec(
            kind="udvf_slicewise",
            weight=1.0,
            options={"num_slices": 5, "direction": "x"},
        )
    if kind == "udvf_piecewise":
        return MetricSpec(
            kind="udvf_piecewise",
            weight=1.0,
            options={"num_pieces": 4, "direction": "x"},
        )
    return MetricSpec(kind=kind, weight=1.0)


def _parameterisation_detail_fields(
    project: IdentificationProject,
    spec: ParameterisationSpec,
    phase_index: int,
) -> list[DetailField]:
    if spec.kind == "known":
        return [
            DetailField(
                key="info",
                label="Info",
                value="No options. Value comes from the constitutive-law table.",
                editor_kind="readonly",
            )
        ]

    if spec.kind == "homogeneous":
        choices = ("initial_value",)
        if phase_index > 0:
            choices = ("initial_value", "previous_phase_result_mean")
        current_value = str(spec.options.get("initialise_from", "initial_value"))
        if current_value == "previous_mean":
            current_value = "previous_phase_result_mean"
        if current_value not in choices:
            current_value = "initial_value"
        return [
            DetailField(
                key="initialise_from",
                label="Initialise from",
                value=current_value,
                editor_kind="combo",
                choices=choices,
            )
        ]

    if spec.kind == "mesh":
        initial_size = spec.options.get("initial_size", [2, 2])
        return [
            DetailField(
                key="initial_size",
                label="Initial size",
                value=f"{initial_size[0]},{initial_size[1]}",
                placeholder="nx,ny",
            ),
            DetailField(
                key="element_order",
                label="Element order",
                value=str(spec.options.get("element_order", 0)),
                editor_kind="combo",
                choices=("0", "1", "2"),
            ),
        ]

    if spec.kind == "basis_function":
        initial_count = spec.options.get("initial_count")
        if initial_count is None:
            initial_count = max(1, len(spec.options.get("kernels", [])))
        return [
            DetailField(
                key="kernel_shape",
                label="Kernel shape",
                value=str(spec.options.get("kernel_shape", "univariate")),
                editor_kind="combo",
                choices=("univariate", "bivariate"),
            ),
            DetailField(
                key="initial_count",
                label="Initial count",
                value=str(initial_count),
                placeholder="number of kernels",
            ),
            DetailField(
                key="addition_method",
                label="Addition method",
                value=str(spec.options.get("addition_method", "place")),
                editor_kind="combo",
                choices=("place", "split", "place_multistart", "none"),
            ),
            DetailField(
                key="optimisation_strategy",
                label="Optimisation",
                value=str(spec.options.get("optimisation_strategy", "all_together")),
                editor_kind="combo",
                choices=("all_together", "fix_centres", "two_stage"),
            ),
        ]

    if spec.kind == "slice_wise":
        return [
            DetailField(
                key="num_slices",
                label="Number of slices",
                value=str(spec.options.get("num_slices", 5)),
            ),
            DetailField(
                key="direction",
                label="Direction",
                value=str(spec.options.get("direction", "x")),
                editor_kind="combo",
                choices=("x", "y"),
            ),
        ]

    if spec.kind == "linked":
        phase_choices = tuple([""] + [phase.name for phase in project.phases])
        return [
            DetailField(
                key="source_phase",
                label="Source phase",
                value=spec.source_phase or "",
                editor_kind="combo",
                choices=phase_choices,
            ),
            DetailField(
                key="source_parameter",
                label="Source parameter",
                value=spec.source_parameter or "",
                editor_kind="combo",
                choices=tuple(project.parameters.keys()),
            ),
            DetailField(
                key="free_dof_groups",
                label="Free DOF groups",
                value=",".join(spec.free_dof_groups or ["value"]),
                placeholder="value,rbf_heights",
            ),
        ]

    return []


def _metric_detail_fields(metric: MetricSpec) -> list[DetailField]:
    if metric.kind == "sbvf":
        mesh_size = metric.options.get("virtual_mesh_size", [15, 15])
        return [
            DetailField(
                key="virtual_mesh_size",
                label="Virtual mesh size",
                value=f"{mesh_size[0]},{mesh_size[1]}",
                placeholder="nx,ny",
            ),
            DetailField(
                key="stress_sensitivity",
                label="Sensitivity map",
                value=str(metric.options.get("stress_sensitivity", "total")),
                editor_kind="combo",
                choices=("total", "incremental"),
            ),
            DetailField(
                key="perturb_type",
                label="Perturb type",
                value=str(metric.options.get("perturb_type", "dof")),
                editor_kind="combo",
                choices=("dof", "parameter"),
            ),
            DetailField(
                key="perturbation_factor",
                label="Perturb fraction",
                value=_format_scalar(metric.options.get("perturbation_factor", 0.15)),
            ),
        ]

    if metric.kind == "egi":
        return [
            DetailField(
                key="window_sizes",
                label="Window sizes",
                value=",".join(str(value) for value in metric.options.get("window_sizes", [5, 9, 13])),
                placeholder="5,9,13",
            ),
            DetailField(
                key="num_windows",
                label="Number of windows",
                value=str(metric.options.get("num_windows", 3)),
            ),
        ]

    if metric.kind == "fre":
        return [
            DetailField(
                key="points_per_slice",
                label="Points per slice",
                value=str(metric.options.get("points_per_slice", 20)),
            ),
            DetailField(
                key="direction",
                label="Direction",
                value=str(metric.options.get("direction", "x")),
                editor_kind="combo",
                choices=("x", "y"),
            ),
        ]

    if metric.kind == "udvf_uniform":
        return [
            DetailField(
                key="info",
                label="Info",
                value="No extra options currently exposed.",
                editor_kind="readonly",
            )
        ]

    if metric.kind == "udvf_slicewise":
        return [
            DetailField(
                key="num_slices",
                label="Number of slices",
                value=str(metric.options.get("num_slices", 5)),
            ),
            DetailField(
                key="direction",
                label="Direction",
                value=str(metric.options.get("direction", "x")),
                editor_kind="combo",
                choices=("x", "y"),
            ),
        ]

    if metric.kind == "udvf_piecewise":
        return [
            DetailField(
                key="num_pieces",
                label="Number of pieces",
                value=str(metric.options.get("num_pieces", 4)),
            ),
            DetailField(
                key="direction",
                label="Direction",
                value=str(metric.options.get("direction", "x")),
                editor_kind="combo",
                choices=("x", "y"),
            ),
        ]

    return []


def _optimiser_detail_fields(optimiser: OptimiserSpec) -> list[DetailField]:
    if optimiser.kind == "least_squares":
        return [
            DetailField(
                key="method",
                label="Method",
                value=str(optimiser.options.get("method", "lm")),
                editor_kind="combo",
                choices=("lm", "trf", "dogbox"),
            ),
            DetailField(
                key="max_nfev",
                label="Max evaluations",
                value=str(optimiser.options.get("max_nfev", 200)),
            ),
        ]

    if optimiser.kind == "pattern_search":
        return [
            DetailField(
                key="max_evaluations",
                label="Max evaluations",
                value=str(optimiser.options.get("max_evaluations", 200)),
            ),
            DetailField(
                key="seed",
                label="Seed",
                value=str(optimiser.options.get("seed", 1)),
            ),
        ]

    return []


class ToolkitWindow(QtWidgets.QWidget):
    """Qt editor for the YAML-backed VFM project model."""

    def __init__(self, project: IdentificationProject) -> None:
        super().__init__()
        self.project = project
        self.project.use_gui = True
        self._renumber_phases()
        self._updating_ui = False
        self._selected_parameterisation_ref: ParameterisationRowRef | None = None
        self._selected_metric_index: int | None = None
        self.setWindowTitle("pyvale VFM Toolkit")
        self.resize(1350, 850)
        self._build_layout()
        self._refresh_all()

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_law_tab(), "Constitutive Law")
        self.tabs.addTab(self._build_phase_tab(), "Identification Phases")
        self.tabs.addTab(self._build_project_tab(), "Project")
        layout.addWidget(self.tabs)

    def _build_law_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        widget.setLayout(layout)

        law_row = QtWidgets.QHBoxLayout()
        law_row.addWidget(QtWidgets.QLabel("Active law"))
        self.law_combo = QtWidgets.QComboBox()
        for law in ConstituitiveLaw:
            self.law_combo.addItem(law.name)
        self.law_combo.currentTextChanged.connect(self._on_law_changed)
        law_row.addWidget(self.law_combo)
        law_row.addStretch(1)
        layout.addLayout(law_row)

        self.parameter_table = QtWidgets.QTableWidget(0, 5)
        self.parameter_table.setHorizontalHeaderLabels(
            ["Parameter", "Initial value type", "Initial value", "Lower", "Upper"]
        )
        self.parameter_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_table.cellChanged.connect(self._on_parameter_cell_changed)
        layout.addWidget(self.parameter_table)

        note_label = QtWidgets.QLabel(
            "If an `.npy` path is provided, the binary should contain a 2D spatial "
            "array matching the specimen coordinate shape.\n"
            "Lower and upper bounds are only used when the parameter is identified "
            "in a phase."
        )
        note_label.setWordWrap(True)
        layout.addWidget(note_label)
        return widget

    def _build_phase_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        widget.setLayout(layout)

        button_row = QtWidgets.QHBoxLayout()
        add_phase_button = QtWidgets.QPushButton("Add Phase")
        add_phase_button.clicked.connect(self._on_add_phase)
        button_row.addWidget(add_phase_button)
        remove_phase_button = QtWidgets.QPushButton("Remove Phase")
        remove_phase_button.clicked.connect(self._on_remove_phase)
        button_row.addWidget(remove_phase_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)

        self.phase_list = QtWidgets.QListWidget()
        self.phase_list.currentRowChanged.connect(self._on_phase_selected)
        splitter.addWidget(self.phase_list)

        editor_widget = QtWidgets.QWidget()
        editor_layout = QtWidgets.QVBoxLayout()
        editor_widget.setLayout(editor_layout)

        editor_layout.addWidget(self._build_parameterisation_group())
        editor_layout.addWidget(self._build_metric_group())
        editor_layout.addWidget(self._build_optimiser_group())
        editor_layout.addStretch(1)

        splitter.addWidget(editor_widget)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        return widget

    def _build_parameterisation_group(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Parameterisations")
        layout = QtWidgets.QVBoxLayout()
        group.setLayout(layout)

        control_row = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton("Add Row Below")
        add_button.clicked.connect(self._on_add_parameterisation_row)
        control_row.addWidget(add_button)
        remove_button = QtWidgets.QPushButton("Delete Current Row")
        remove_button.clicked.connect(self._on_remove_parameterisation_row)
        control_row.addWidget(remove_button)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        pair_splitter = QtWidgets.QSplitter()
        pair_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)

        overview_widget = QtWidgets.QWidget()
        overview_layout = QtWidgets.QVBoxLayout()
        overview_widget.setLayout(overview_layout)

        self.parameterisation_overview = QtWidgets.QTableWidget(0, 2)
        self.parameterisation_overview.setHorizontalHeaderLabels(
            ["Parameter", "Type"]
        )
        self.parameterisation_overview.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.parameterisation_overview.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.parameterisation_overview.itemSelectionChanged.connect(
            self._on_parameterisation_selection_changed
        )
        self.parameterisation_overview.horizontalHeader().setStretchLastSection(True)
        overview_layout.addWidget(self.parameterisation_overview)
        pair_splitter.addWidget(overview_widget)

        detail_widget = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout()
        detail_widget.setLayout(detail_layout)

        self.parameterisation_detail = QtWidgets.QTableWidget(0, 2)
        self.parameterisation_detail.setHorizontalHeaderLabels(["Option", "Value"])
        self.parameterisation_detail.horizontalHeader().setStretchLastSection(True)
        detail_layout.addWidget(self.parameterisation_detail)
        pair_splitter.addWidget(detail_widget)
        pair_splitter.setStretchFactor(0, 2)
        pair_splitter.setStretchFactor(1, 3)
        layout.addWidget(pair_splitter)
        return group

    def _build_metric_group(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Cost Function / Metrics")
        layout = QtWidgets.QVBoxLayout()
        group.setLayout(layout)

        control_row = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton("Add Metric")
        add_button.clicked.connect(self._on_add_metric)
        control_row.addWidget(add_button)
        remove_button = QtWidgets.QPushButton("Remove Metric")
        remove_button.clicked.connect(self._on_remove_metric)
        control_row.addWidget(remove_button)
        control_row.addStretch(1)
        self.metric_weight_label = QtWidgets.QLabel()
        control_row.addWidget(self.metric_weight_label)
        layout.addLayout(control_row)

        pair_splitter = QtWidgets.QSplitter()
        pair_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)

        overview_widget = QtWidgets.QWidget()
        overview_layout = QtWidgets.QVBoxLayout()
        overview_widget.setLayout(overview_layout)

        self.metric_overview = QtWidgets.QTableWidget(0, 2)
        self.metric_overview.setHorizontalHeaderLabels(["Metric", "Weight"])
        self.metric_overview.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.metric_overview.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.metric_overview.itemSelectionChanged.connect(
            self._on_metric_selection_changed
        )
        self.metric_overview.horizontalHeader().setStretchLastSection(True)
        overview_layout.addWidget(self.metric_overview)
        pair_splitter.addWidget(overview_widget)

        detail_widget = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout()
        detail_widget.setLayout(detail_layout)

        self.metric_detail = QtWidgets.QTableWidget(0, 2)
        self.metric_detail.setHorizontalHeaderLabels(["Option", "Value"])
        self.metric_detail.horizontalHeader().setStretchLastSection(True)
        detail_layout.addWidget(self.metric_detail)
        pair_splitter.addWidget(detail_widget)
        pair_splitter.setStretchFactor(0, 2)
        pair_splitter.setStretchFactor(1, 3)
        layout.addWidget(pair_splitter)
        return group

    def _build_optimiser_group(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Optimiser")
        layout = QtWidgets.QVBoxLayout()
        group.setLayout(layout)

        pair_splitter = QtWidgets.QSplitter()
        pair_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)

        overview_widget = QtWidgets.QWidget()
        overview_layout = QtWidgets.QVBoxLayout()
        overview_widget.setLayout(overview_layout)

        self.optimiser_overview = QtWidgets.QTableWidget(1, 1)
        self.optimiser_overview.setHorizontalHeaderLabels(["Type"])
        self.optimiser_overview.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.optimiser_overview.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.optimiser_overview.horizontalHeader().setStretchLastSection(True)
        self.optimiser_overview.itemSelectionChanged.connect(
            self._on_optimiser_selection_changed
        )
        overview_layout.addWidget(self.optimiser_overview)
        pair_splitter.addWidget(overview_widget)

        detail_widget = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout()
        detail_widget.setLayout(detail_layout)

        self.optimiser_detail = QtWidgets.QTableWidget(0, 2)
        self.optimiser_detail.setHorizontalHeaderLabels(["Option", "Value"])
        self.optimiser_detail.horizontalHeader().setStretchLastSection(True)
        detail_layout.addWidget(self.optimiser_detail)
        pair_splitter.addWidget(detail_widget)
        pair_splitter.setStretchFactor(0, 2)
        pair_splitter.setStretchFactor(1, 3)
        layout.addWidget(pair_splitter)
        return group

    def _build_project_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        widget.setLayout(layout)

        form_layout = QtWidgets.QGridLayout()
        form_layout.setColumnStretch(1, 1)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(10)

        self.project_name_edit = QtWidgets.QLineEdit()
        self.project_name_edit.editingFinished.connect(self._on_project_name_changed)
        form_layout.addWidget(QtWidgets.QLabel("Project name"), 0, 0)
        form_layout.addWidget(self.project_name_edit, 0, 1)

        path_row = QtWidgets.QHBoxLayout()
        self.test_data_path_edit = QtWidgets.QLineEdit()
        self.test_data_path_edit.editingFinished.connect(self._on_test_data_path_changed)
        path_row.addWidget(self.test_data_path_edit)
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._on_browse_test_data)
        path_row.addWidget(browse_button)
        path_widget = QtWidgets.QWidget()
        path_widget.setLayout(path_row)
        form_layout.addWidget(QtWidgets.QLabel("Test data"), 1, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        form_layout.addWidget(path_widget, 1, 1)

        self.project_notes_edit = QtWidgets.QPlainTextEdit()
        self.project_notes_edit.textChanged.connect(self._on_project_notes_changed)
        line_height = self.project_notes_edit.fontMetrics().lineSpacing()
        self.project_notes_edit.setFixedHeight((2 * line_height) + 16)
        form_layout.addWidget(QtWidgets.QLabel("Notes"), 2, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        form_layout.addWidget(self.project_notes_edit, 2, 1)
        layout.addLayout(form_layout)

        layout.addWidget(QtWidgets.QLabel("Project preview"))
        self.project_preview = QtWidgets.QPlainTextEdit()
        self.project_preview.setReadOnly(True)
        layout.addWidget(self.project_preview)

        button_row = QtWidgets.QHBoxLayout()
        load_button = QtWidgets.QPushButton("Load Project YAML")
        load_button.clicked.connect(self._load_project)
        button_row.addWidget(load_button)
        save_button = QtWidgets.QPushButton("Save Project YAML")
        save_button.clicked.connect(self._save_project)
        button_row.addWidget(save_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return widget

    def _refresh_all(self) -> None:
        self._refresh_law_tab()
        self._refresh_phase_list()
        self._refresh_phase_editor()
        self._refresh_project_tab()
        self._refresh_project_preview()

    def _refresh_law_tab(self) -> None:
        self._updating_ui = True
        try:
            self.law_combo.setCurrentText(self.project.constituitive_law.name)
            parameter_names = list(self.project.parameters.keys())
            self.parameter_table.blockSignals(True)
            self.parameter_table.setRowCount(len(parameter_names))
            for row, parameter_name in enumerate(parameter_names):
                parameter_definition = self.project.parameters[parameter_name]
                self.parameter_table.setItem(row, 0, _readonly_item(parameter_name))
                type_combo = QtWidgets.QComboBox()
                for initial_value_type in INITIAL_VALUE_TYPES:
                    type_combo.addItem(initial_value_type)
                type_combo.setCurrentText(parameter_definition.initial_value_type)
                type_combo.currentTextChanged.connect(
                    lambda value, parameter_name=parameter_name: self._on_initial_value_type_changed(
                        parameter_name,
                        value,
                    )
                )
                self.parameter_table.setCellWidget(row, 1, type_combo)
                self.parameter_table.setItem(
                    row, 2, QtWidgets.QTableWidgetItem(_format_initial_value(parameter_definition.initial_value))
                )
                self.parameter_table.setItem(
                    row, 3, QtWidgets.QTableWidgetItem(_format_scalar(parameter_definition.lower_bound))
                )
                self.parameter_table.setItem(
                    row, 4, QtWidgets.QTableWidgetItem(_format_scalar(parameter_definition.upper_bound))
                )
            self.parameter_table.blockSignals(False)
        finally:
            self._updating_ui = False

    def _refresh_phase_list(self) -> None:
        selected_row = self.phase_list.currentRow()
        self.phase_list.blockSignals(True)
        self.phase_list.clear()
        for phase in self.project.phases:
            self.phase_list.addItem(phase.name)
        self.phase_list.blockSignals(False)

        if self.project.phases:
            next_row = min(max(selected_row, 0), len(self.project.phases) - 1)
            self.phase_list.setCurrentRow(next_row)
        else:
            self.phase_list.setCurrentRow(-1)

    def _refresh_phase_editor(self) -> None:
        phase = self._current_phase()
        enabled = phase is not None

        self.parameterisation_overview.setEnabled(enabled)
        self.parameterisation_detail.setEnabled(enabled)
        self.metric_overview.setEnabled(enabled)
        self.metric_detail.setEnabled(enabled)
        self.optimiser_overview.setEnabled(enabled)
        self.optimiser_detail.setEnabled(enabled)

        if phase is None:
            self.parameterisation_overview.setRowCount(0)
            self.parameterisation_detail.setRowCount(0)
            self.metric_overview.setRowCount(0)
            self.metric_detail.setRowCount(0)
            self.optimiser_overview.setRowCount(1)
            self.optimiser_detail.setRowCount(0)
            self.metric_weight_label.setText("")
            return

        self._ensure_phase_defaults(phase)
        self._updating_ui = True
        try:
            self._refresh_parameterisation_section(phase)
            self._refresh_metric_section(phase)
            self._refresh_optimiser_section(phase)
        finally:
            self._updating_ui = False

    def _refresh_parameterisation_section(self, phase: PhaseDefinition) -> None:
        rows = _parameterisation_rows(phase)
        selected_ref = self._resolve_selected_parameterisation_ref(rows)

        self.parameterisation_overview.blockSignals(True)
        self.parameterisation_overview.setRowCount(len(rows))
        for row_index, row_ref in enumerate(rows):
            spec = phase.parameterisations[row_ref.parameter_name][row_ref.spec_index]
            self.parameterisation_overview.setItem(
                row_index,
                0,
                _readonly_item(row_ref.parameter_name),
            )

            kind_combo = QtWidgets.QComboBox()
            for kind in PARAMETERISATION_KINDS:
                kind_combo.addItem(kind)
            kind_combo.setCurrentText(spec.kind)
            kind_combo.currentTextChanged.connect(
                lambda kind, row_ref=row_ref: self._on_parameterisation_kind_changed(
                    row_ref,
                    kind,
                )
            )
            self.parameterisation_overview.setCellWidget(row_index, 1, kind_combo)
        self.parameterisation_overview.blockSignals(False)

        if rows:
            selected_row = rows.index(selected_ref)
            self.parameterisation_overview.selectRow(selected_row)
            self._selected_parameterisation_ref = selected_ref
        else:
            self._selected_parameterisation_ref = None

        self._refresh_parameterisation_detail_table()

    def _refresh_metric_section(self, phase: PhaseDefinition) -> None:
        if not phase.metrics:
            phase.metrics = [MetricSpec(kind="sbvf", weight=1.0)]

        self.metric_weight_label.setText(
            f"Current weight sum: {sum(metric.weight for metric in phase.metrics):.6g}"
        )

        if self._selected_metric_index is None or self._selected_metric_index >= len(phase.metrics):
            self._selected_metric_index = 0

        self.metric_overview.blockSignals(True)
        self.metric_overview.setRowCount(len(phase.metrics))
        for row_index, metric in enumerate(phase.metrics):
            kind_combo = QtWidgets.QComboBox()
            for kind in METRIC_KINDS:
                kind_combo.addItem(kind)
            kind_combo.setCurrentText(metric.kind)
            kind_combo.currentTextChanged.connect(
                lambda kind, row_index=row_index: self._on_metric_kind_changed(
                    row_index,
                    kind,
                )
            )
            self.metric_overview.setCellWidget(row_index, 0, kind_combo)

            weight_spin = QtWidgets.QDoubleSpinBox()
            weight_spin.setDecimals(6)
            weight_spin.setRange(0.0, 1.0e6)
            weight_spin.setValue(metric.weight)
            weight_spin.valueChanged.connect(
                lambda value, row_index=row_index: self._on_metric_weight_changed(
                    row_index,
                    value,
                )
            )
            self.metric_overview.setCellWidget(row_index, 1, weight_spin)
        self.metric_overview.blockSignals(False)
        self.metric_overview.selectRow(self._selected_metric_index)

        self._refresh_metric_detail_table()

    def _refresh_optimiser_section(self, phase: PhaseDefinition) -> None:
        self.optimiser_overview.blockSignals(True)
        self.optimiser_overview.setRowCount(1)

        optimiser_combo = QtWidgets.QComboBox()
        for kind in OPTIMISER_KINDS:
            optimiser_combo.addItem(kind)
        optimiser_combo.setCurrentText(phase.optimiser.kind)
        optimiser_combo.currentTextChanged.connect(self._on_optimiser_kind_changed)
        self.optimiser_overview.setCellWidget(0, 0, optimiser_combo)
        self.optimiser_overview.blockSignals(False)
        self.optimiser_overview.selectRow(0)

        self._refresh_optimiser_detail_table()

    def _refresh_parameterisation_detail_table(self) -> None:
        phase = self._current_phase()
        row_ref = self._selected_parameterisation_ref
        if phase is None or row_ref is None:
            self.parameterisation_detail.setRowCount(0)
            return

        spec = phase.parameterisations[row_ref.parameter_name][row_ref.spec_index]
        fields = _parameterisation_detail_fields(
            self.project,
            spec,
            self.phase_list.currentRow(),
        )
        self._populate_detail_table(
            self.parameterisation_detail,
            fields,
            self._on_parameterisation_detail_changed,
        )

    def _refresh_metric_detail_table(self) -> None:
        phase = self._current_phase()
        if phase is None or self._selected_metric_index is None:
            self.metric_detail.setRowCount(0)
            return

        if self._selected_metric_index >= len(phase.metrics):
            self._selected_metric_index = 0
        metric = phase.metrics[self._selected_metric_index]
        fields = _metric_detail_fields(metric)
        self._populate_detail_table(
            self.metric_detail,
            fields,
            self._on_metric_detail_changed,
        )

    def _refresh_optimiser_detail_table(self) -> None:
        phase = self._current_phase()
        if phase is None:
            self.optimiser_detail.setRowCount(0)
            return

        fields = _optimiser_detail_fields(phase.optimiser)
        self._populate_detail_table(
            self.optimiser_detail,
            fields,
            self._on_optimiser_detail_changed,
        )

    def _populate_detail_table(
        self,
        table: QtWidgets.QTableWidget,
        fields: list[DetailField],
        change_handler,
    ) -> None:
        table.blockSignals(True)
        table.setRowCount(len(fields))
        for row_index, field in enumerate(fields):
            table.setItem(row_index, 0, _readonly_item(field.label))

            if field.editor_kind == "readonly":
                table.setItem(row_index, 1, _readonly_item(field.value))
                continue

            if field.editor_kind == "combo":
                combo = QtWidgets.QComboBox()
                for choice in field.choices:
                    combo.addItem(choice)
                combo.setCurrentText(field.value)
                combo.currentTextChanged.connect(
                    lambda value, key=field.key: change_handler(key, value)
                )
                table.setCellWidget(row_index, 1, combo)
                continue

            line_edit = QtWidgets.QLineEdit(field.value)
            line_edit.setPlaceholderText(field.placeholder)
            line_edit.editingFinished.connect(
                lambda key=field.key, line_edit=line_edit: change_handler(
                    key,
                    line_edit.text(),
                )
            )
            table.setCellWidget(row_index, 1, line_edit)
        table.blockSignals(False)

    def _resolve_selected_parameterisation_ref(
        self,
        rows: list[ParameterisationRowRef],
    ) -> ParameterisationRowRef:
        if rows and self._selected_parameterisation_ref in rows:
            return self._selected_parameterisation_ref
        return rows[0]

    def _current_phase(self) -> PhaseDefinition | None:
        row = self.phase_list.currentRow()
        if row < 0 or row >= len(self.project.phases):
            return None
        return self.project.phases[row]

    def _renumber_phases(self) -> None:
        for phase_index, phase in enumerate(self.project.phases, start=1):
            phase.name = f"phase_{phase_index}"

    def _ensure_phase_defaults(self, phase: PhaseDefinition) -> None:
        phase_index = self.project.phases.index(phase)
        for parameter_name in self.project.parameters:
            if parameter_name not in phase.parameterisations or not phase.parameterisations[parameter_name]:
                phase.parameterisations[parameter_name] = [
                    _default_parameterisation_spec(self.project.constituitive_law, parameter_name)
                ]
            for spec in phase.parameterisations[parameter_name]:
                if spec.kind != "homogeneous":
                    continue
                if spec.options.get("initialise_from") == "previous_mean":
                    spec.options["initialise_from"] = "previous_phase_result_mean"
                if phase_index == 0 and spec.options.get("initialise_from") == "previous_phase_result_mean":
                    spec.options["initialise_from"] = "initial_value"

        if not phase.metrics:
            phase.metrics = [MetricSpec(kind="sbvf", weight=1.0)]

    def _show_message(self, title: str, text: str) -> None:
        QtWidgets.QMessageBox.information(self, title, text)

    def _on_law_changed(self, law_name: str) -> None:
        if self._updating_ui:
            return

        constituitive_law = ConstituitiveLaw[law_name]
        if constituitive_law is self.project.constituitive_law:
            return

        self.project.constituitive_law = constituitive_law
        self.project.parameters = {
            parameter_name.name: create_default_parameter_definition(
                constituitive_law,
                parameter_name,
            )
            for parameter_name in required_parameters_for_law(constituitive_law)
        }
        self.project.phases = [create_default_phase_definition(constituitive_law)]
        self._selected_parameterisation_ref = None
        self._selected_metric_index = 0
        self._refresh_all()

    def _on_parameter_cell_changed(self, row: int, column: int) -> None:
        if self._updating_ui or column in {0, 1}:
            return

        parameter_name = self.parameter_table.item(row, 0).text()
        parameter_definition = self.project.parameters[parameter_name]
        cell_item = self.parameter_table.item(row, column)
        text = "" if cell_item is None else cell_item.text()

        if column == 2:
            if parameter_definition.initial_value_type == "2d np array":
                parameter_definition.initial_value = text.strip()
            else:
                parameter_definition.initial_value = _parse_float(text, None)
        elif column == 3:
            parameter_definition.lower_bound = _parse_float(text, parameter_definition.lower_bound)
        elif column == 4:
            parameter_definition.upper_bound = _parse_float(text, parameter_definition.upper_bound)

        self._refresh_law_tab()
        self._refresh_project_preview()

    def _on_initial_value_type_changed(self, parameter_name: str, initial_value_type: str) -> None:
        if self._updating_ui:
            return

        parameter_definition = self.project.parameters[parameter_name]
        if parameter_definition.initial_value_type == initial_value_type:
            return

        parameter_definition.initial_value_type = initial_value_type
        if initial_value_type == "2d np array":
            parameter_definition.initial_value = ""
        else:
            default_parameter = create_default_parameter_definition(
                self.project.constituitive_law,
                parameter_definition.name,
            )
            parameter_definition.initial_value = default_parameter.initial_value

        self._refresh_law_tab()
        self._refresh_project_preview()

    def _on_add_phase(self) -> None:
        phase = create_default_phase_definition(
            self.project.constituitive_law,
            phase_index=len(self.project.phases) + 1,
        )
        self.project.phases.append(phase)
        self._renumber_phases()
        self._selected_parameterisation_ref = None
        self._selected_metric_index = 0
        self._refresh_phase_list()
        self.phase_list.setCurrentRow(len(self.project.phases) - 1)
        self._refresh_project_preview()

    def _on_remove_phase(self) -> None:
        row = self.phase_list.currentRow()
        if row < 0:
            return

        self.project.phases.pop(row)
        self._renumber_phases()
        self._selected_parameterisation_ref = None
        self._selected_metric_index = 0
        self._refresh_phase_list()
        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_phase_selected(self, _: int) -> None:
        self._selected_parameterisation_ref = None
        self._selected_metric_index = 0
        self._refresh_phase_editor()

    def _on_parameterisation_selection_changed(self) -> None:
        if self._updating_ui:
            return

        phase = self._current_phase()
        if phase is None:
            return

        selected_rows = self.parameterisation_overview.selectionModel().selectedRows()
        if not selected_rows:
            return

        rows = _parameterisation_rows(phase)
        selected_row = selected_rows[0].row()
        if 0 <= selected_row < len(rows):
            self._selected_parameterisation_ref = rows[selected_row]
            self._refresh_parameterisation_detail_table()

    def _on_add_parameterisation_row(self) -> None:
        phase = self._current_phase()
        if phase is None:
            return

        row_ref = self._selected_parameterisation_ref
        if row_ref is None:
            parameter_name = next(iter(self.project.parameters))
            insert_index = len(phase.parameterisations.setdefault(parameter_name, []))
        else:
            parameter_name = row_ref.parameter_name
            insert_index = row_ref.spec_index + 1

        specs = phase.parameterisations.setdefault(parameter_name, [])
        if any(spec.kind in EXCLUSIVE_PARAMETERISATION_KINDS for spec in specs):
            self._show_message(
                "Cannot Add Row",
                f"Parameter '{parameter_name}' currently uses an exclusive "
                "parameterisation (`known` or `linked`). Change that row first "
                "before stacking additional parameterisations.",
            )
            return

        specs.insert(insert_index, _stackable_parameterisation_spec(parameter_name))
        self._selected_parameterisation_ref = ParameterisationRowRef(
            parameter_name=parameter_name,
            spec_index=insert_index,
        )
        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_remove_parameterisation_row(self) -> None:
        phase = self._current_phase()
        row_ref = self._selected_parameterisation_ref
        if phase is None or row_ref is None:
            return

        specs = phase.parameterisations[row_ref.parameter_name]
        if len(specs) <= 1:
            self._show_message(
                "Cannot Remove Row",
                "Each parameter needs at least one parameterisation row in the phase.",
            )
            return

        specs.pop(row_ref.spec_index)
        new_index = max(0, row_ref.spec_index - 1)
        self._selected_parameterisation_ref = ParameterisationRowRef(
            parameter_name=row_ref.parameter_name,
            spec_index=new_index,
        )
        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_parameterisation_detail_changed(self, key: str, value: str) -> None:
        if self._updating_ui:
            return

        phase = self._current_phase()
        row_ref = self._selected_parameterisation_ref
        if phase is None or row_ref is None:
            return

        spec = phase.parameterisations[row_ref.parameter_name][row_ref.spec_index]

        if key == "initialise_from":
            if self.phase_list.currentRow() == 0 and value == "previous_phase_result_mean":
                value = "initial_value"
            spec.options["initialise_from"] = value
        elif key == "initial_size":
            spec.options["initial_size"] = _parse_int_pair(value, [2, 2])
        elif key == "element_order":
            spec.options["element_order"] = _parse_int(value, 0)
        elif key == "kernel_shape":
            spec.options["kernel_shape"] = value
        elif key == "initial_count":
            spec.options["initial_count"] = max(1, _parse_int(value, 1))
        elif key == "addition_method":
            spec.options["addition_method"] = value
        elif key == "optimisation_strategy":
            spec.options["optimisation_strategy"] = value
        elif key == "num_slices":
            spec.options["num_slices"] = max(1, _parse_int(value, 5))
        elif key == "direction":
            spec.options["direction"] = value
        elif key == "source_phase":
            spec.source_phase = value or None
        elif key == "source_parameter":
            spec.source_parameter = value or None
        elif key == "free_dof_groups":
            spec.free_dof_groups = _parse_csv(value, ["value"])

        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_parameterisation_kind_changed(
        self,
        row_ref: ParameterisationRowRef,
        kind: str,
    ) -> None:
        if self._updating_ui:
            return

        phase = self._current_phase()
        if phase is None:
            return

        specs = phase.parameterisations[row_ref.parameter_name]
        current_spec = specs[row_ref.spec_index]
        if current_spec.kind == kind:
            self._selected_parameterisation_ref = row_ref
            self._refresh_parameterisation_detail_table()
            return

        if kind in EXCLUSIVE_PARAMETERISATION_KINDS and len(specs) > 1:
            self._show_message(
                "Exclusive Parameterisation",
                f"`{kind}` cannot be combined with other parameterisation rows "
                f"for '{row_ref.parameter_name}'. Remove the extra rows first.",
            )
            self._refresh_phase_editor()
            return

        specs[row_ref.spec_index] = _parameterisation_spec_for_kind(
            row_ref.parameter_name,
            kind,
        )
        self._selected_parameterisation_ref = row_ref
        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_metric_selection_changed(self) -> None:
        if self._updating_ui:
            return

        selected_rows = self.metric_overview.selectionModel().selectedRows()
        if not selected_rows:
            return

        self._selected_metric_index = selected_rows[0].row()
        self._refresh_metric_detail_table()

    def _on_add_metric(self) -> None:
        phase = self._current_phase()
        if phase is None:
            return

        phase.metrics.append(
            MetricSpec(
                kind="sbvf",
                weight=1.0,
                options={
                    "virtual_mesh_size": [15, 15],
                    "stress_sensitivity": "total",
                    "perturb_type": "dof",
                    "perturbation_factor": 0.15,
                },
            )
        )
        self._selected_metric_index = len(phase.metrics) - 1
        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_remove_metric(self) -> None:
        phase = self._current_phase()
        if phase is None or self._selected_metric_index is None:
            return

        if len(phase.metrics) <= 1:
            self._show_message(
                "Cannot Remove Metric",
                "Each phase needs at least one cost-function / metric row.",
            )
            return

        phase.metrics.pop(self._selected_metric_index)
        self._selected_metric_index = max(0, self._selected_metric_index - 1)
        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_metric_weight_changed(self, metric_index: int, value: float) -> None:
        if self._updating_ui:
            return

        phase = self._current_phase()
        if phase is None or metric_index >= len(phase.metrics):
            return

        phase.metrics[metric_index].weight = float(value)
        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_metric_kind_changed(self, metric_index: int, kind: str) -> None:
        if self._updating_ui:
            return

        phase = self._current_phase()
        if phase is None or metric_index >= len(phase.metrics):
            return

        previous_weight = phase.metrics[metric_index].weight
        phase.metrics[metric_index] = _metric_spec_for_kind(kind)
        phase.metrics[metric_index].weight = previous_weight
        self._selected_metric_index = metric_index
        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_metric_detail_changed(self, key: str, value: str) -> None:
        if self._updating_ui:
            return

        phase = self._current_phase()
        if phase is None or self._selected_metric_index is None:
            return

        metric = phase.metrics[self._selected_metric_index]

        if key == "virtual_mesh_size":
            metric.options["virtual_mesh_size"] = _parse_int_pair(value, [15, 15])
        elif key == "stress_sensitivity":
            metric.options["stress_sensitivity"] = value
        elif key == "perturb_type":
            metric.options["perturb_type"] = value
        elif key == "perturbation_factor":
            metric.options["perturbation_factor"] = _parse_float(value, 0.15)
        elif key == "window_sizes":
            metric.options["window_sizes"] = _parse_number_list(value, [5, 9, 13])
        elif key == "num_windows":
            metric.options["num_windows"] = max(1, _parse_int(value, 3))
        elif key == "points_per_slice":
            metric.options["points_per_slice"] = max(1, _parse_int(value, 20))
        elif key == "direction":
            metric.options["direction"] = value
        elif key == "num_slices":
            metric.options["num_slices"] = max(1, _parse_int(value, 5))
        elif key == "num_pieces":
            metric.options["num_pieces"] = max(1, _parse_int(value, 4))

        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_optimiser_selection_changed(self) -> None:
        if self._updating_ui:
            return
        self._refresh_optimiser_detail_table()

    def _on_optimiser_kind_changed(self, kind: str) -> None:
        if self._updating_ui:
            return

        phase = self._current_phase()
        if phase is None:
            return

        if kind == "least_squares":
            phase.optimiser = OptimiserSpec(
                kind="least_squares",
                options={"method": "lm", "max_nfev": 200},
            )
        elif kind == "pattern_search":
            phase.optimiser = OptimiserSpec(
                kind="pattern_search",
                options={"max_evaluations": 200, "seed": 1},
            )
        else:
            phase.optimiser = OptimiserSpec(kind=kind)

        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_optimiser_detail_changed(self, key: str, value: str) -> None:
        if self._updating_ui:
            return

        phase = self._current_phase()
        if phase is None:
            return

        if key == "method":
            phase.optimiser.options["method"] = value
        elif key == "max_nfev":
            phase.optimiser.options["max_nfev"] = max(1, _parse_int(value, 200))
        elif key == "max_evaluations":
            phase.optimiser.options["max_evaluations"] = max(1, _parse_int(value, 200))
        elif key == "seed":
            phase.optimiser.options["seed"] = _parse_int(value, 1)

        self._refresh_phase_editor()
        self._refresh_project_preview()

    def _on_project_name_changed(self) -> None:
        if self._updating_ui:
            return
        self.project.name = self.project_name_edit.text().strip() or self.project.name
        self._refresh_project_preview()

    def _on_test_data_path_changed(self) -> None:
        if self._updating_ui:
            return
        text = self.test_data_path_edit.text().strip()
        self.project.test_data_path = Path(text) if text else None
        self._refresh_project_preview()

    def _on_browse_test_data(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Test Data",
            str(self.project.test_data_path) if self.project.test_data_path is not None else "",
            filter="Parsed Test Data (*.npz);;MAT Files (*.mat);;All Files (*)",
        )
        if not filename:
            return

        self.project.test_data_path = Path(filename)
        self._refresh_project_tab()
        self._refresh_project_preview()

    def _on_project_notes_changed(self) -> None:
        if self._updating_ui:
            return
        self.project.notes = self.project_notes_edit.toPlainText()
        self._refresh_project_preview()

    def _refresh_project_tab(self) -> None:
        self._updating_ui = True
        try:
            self.project_name_edit.setText(self.project.name)
            self.test_data_path_edit.setText(
                "" if self.project.test_data_path is None else str(self.project.test_data_path)
            )
            self.project_notes_edit.blockSignals(True)
            self.project_notes_edit.setPlainText(self.project.notes)
            self.project_notes_edit.blockSignals(False)
        finally:
            self._updating_ui = False

    def _refresh_project_preview(self) -> None:
        preview = project_to_yaml_text(self.project)
        self.project_preview.setPlainText(preview)

    def _save_project(self) -> None:
        start_path = (
            str(self.project.project_path)
            if self.project.project_path is not None
            else "vfm_project.yaml"
        )
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save VFM Project",
            start_path,
            filter="YAML Files (*.yaml)",
        )
        if not filename:
            return

        if not filename.endswith(".yaml"):
            filename += ".yaml"

        save_project(self.project, Path(filename))
        self.project.project_path = Path(filename)

    def _load_project(self) -> None:
        start_path = (
            str(self.project.project_path)
            if self.project.project_path is not None
            else "vfm_project.yaml"
        )
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load VFM Project",
            start_path,
            filter="YAML Files (*.yaml);;All Files (*)",
        )
        if not filename:
            return

        loaded_project = load_project(Path(filename))
        self.project = loaded_project
        self.project.use_gui = True
        self._renumber_phases()
        self._selected_parameterisation_ref = None
        self._selected_metric_index = 0
        self._refresh_all()


def launch_gui(project: IdentificationProject) -> IdentificationProject:
    """Open the toolkit GUI and return the edited project."""

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv)

    window = ToolkitWindow(project)
    window.show()

    if owns_app:
        app.exec()

    return project
