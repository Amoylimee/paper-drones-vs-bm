# AGENTS.md

## Project

[Project name and one-line goal]

---

## Domain Context

This template is for projects in maritime data science, including:

- shipping big-data analytics
- machine learning and deep learning for maritime applications
- AIS-based vessel behavior and traffic analysis
- ship energy, fuel, and emissions analysis

Project-specific research questions should be defined per repository, not hardcoded in this template.

---

## Working Style (Author Signature)

This project follows a strict `P` and `p` structure:

- Pipeline folders: `P0_...`, `P1_...`, `P2_...` (uppercase `P`)
- Script files: `p0_...py`, `p1_...py`, `p2_...py` (lowercase `p`)
- Output folders: `output/P{N}/p{n}_{name}/`
- Log folders: `logs/P{N}/p{n}_{name}/`

Do not mix outputs across pipelines. Each pipeline owns its own output and log paths.

---

## Custom Libraries

> Internal libraries. Do not replace with external packages unless explicitly requested.

### `sinbue` (`import sinbue as sb`)

Primary use cases:

- sequential time deltas (`calculate_sequential_time`)
- parallel file processing (`process_files_parallel`)
- point-to-grid aggregation (`get_points_to_grids`)
- run-level logging (`PrintRedirector`)
- activity mode classification (`identify_vessel_activity_mode`)

### `marenem` (`import marenem as mrn`)

Primary use cases:

- engine load and work calculations (`work_ME_Load`, `work_AEAB_Load`, `work_Total_Work`)
- fuel consumption (`fuel_calculate_consumption`)
- pollutant emissions (`em_CO2`, `em_CH4`, `em_N2O`, `em_NOx`, `em_SOx`, `em_CO`, `em_BC`, `em_NMVOC`, `em_PM`)

### `iogenius` (`import iogenius as iog`)

Primary use cases:

- file concatenation (`concat_files_in_folder`)
- directory creation (`create_new_directory`)

---

## Data Conventions

Define and keep stable throughout project:

- ID column: `[e.g., AIS_new]`
- Segment column: `[e.g., segment_id]`
- Time column: `[e.g., UTC]`
- Coordinates: `[e.g., lon, lat]`
- Speed column: `[e.g., speed]`
- Course column: `[e.g., course]`

Recommended emission naming pattern:

- `{Pollutant}_Total_{FuelType}_g`
- Example: `CO2_Total_MDO_g`, `NOx_Total_MDO_g`, `PM2.5_Total_MDO_g`

File format preference:

- Feather (`.feather`) unless project-specific reasons require otherwise

---

## Pipeline Rules

- One objective per pipeline (`P0`, `P1`, `P2`, ...)
- No hidden side effects between pipelines
- Every pipeline should be runnable independently once its inputs exist
- Keep intermediate outputs versionable and inspectable

For analysis/reporting pipelines:

- Tables -> `output/P{N}/p{n}_tables/`
- Figures -> `output/P{N}/p{n}_figures/`
- If `.tex` outputs are added or renamed, update Makefile targets accordingly

---

## Coding Preferences

- Use `Path` objects, not raw string paths
- Set the working directory to the project root in every script to keep workflows portable across computers
- Prefer relative paths when writing code; use absolute paths only when there is no practical alternative
- Use `natsort.natsorted()` for file ordering
- Prefer vectorized Pandas/Numpy operations
- Keep functions small and single-purpose
- Use explicit column lists in transformation functions
- Keep random processes reproducible with fixed seeds

Figures:

- Save both PNG (300 dpi) and PDF

---

## Script Templates

```python
from pathlib import Path
import natsort
import pandas as pd
import numpy as np
import sinbue as sb
import iogenius as iog

if __name__ == "__main__":
    # display all columns
    pd.set_option("display.max_columns", None)
    # display top 100 rows
    pd.set_option("display.max_rows", 100)

    iog.set_working_directory("path_to_working_directory")
    # your code here
```

Parallel-processing template:

```python
from pathlib import Path
import natsort
import sinbue as sb
import iogenius as iog
import pandas as pd
import numpy as np

def main(file_in, output_dir, log_dir):
    file_in = Path(file_in)
    output_dir = Path(output_dir)
    log_dir = Path(log_dir)

    with sb.PrintRedirector(log_dir / f"{file_in.stem}.log"):
        print(f"Processing {file_in}")
        df = pd.read_feather(file_in)

        # processing logic

        df.to_feather(output_dir / f"{file_in.stem}.feather")


if __name__ == "__main__":
    # display all columns
    pd.set_option("display.max_columns", None)
    # display top 100 rows
    pd.set_option("display.max_rows", 100)

    iog.set_working_directory("path_to_working_directory")

    output_dir = Path("./output/PN_name")
    log_dir = Path("./logs/PN_name")
    iog.create_new_directory(output_dir)
    iog.create_new_directory(log_dir)

    # Get the data from the input files
    input_path = Path("path_to_input_data")
    input_files = list(input_path.glob("*.filetype"))
    input_files = natsort.natsorted(input_files)

    sb.process_files_parallel(
        main,
        input_files=input_files,
        output_dir=output_dir,
        log_dir=log_dir,
        max_workers=24,
        show_progress=True,
    )
```

Template selection rule:

- Default to the Basic template.
- Use the Parallel-processing template only when explicitly requested.

---

## Agent Execution Guidance

When an AI coding agent works in this project:

- Run code under conda environment `sinbue_env` to ensure all internal libraries are available
- Preserve `P`/`p` naming style
- Prefer reusing `sinbue` and `marenem` functions before writing new equivalents
- Keep outputs and logs in pipeline-specific directories
- Do not silently rename core columns without a compatibility step
- For major analysis additions, include:
  - one script in the corresponding `P{N}_...` folder
  - one output folder in `output/`
  - one log folder in `logs/`

---

## New Project Bootstrap Checklist

Before starting a new project, replace placeholders in this file:

- Project name and objective
- Fixed column dictionary (if needed)
- Pipeline definitions (`P0`, `P1`, `P2`, ...)
- Output artifact expectations (tables, figures, model files, etc.)

Then save as `AGENTS.md` in the new project root.
