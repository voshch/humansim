from pathlib import Path
import json

from pydantic import BaseModel

BaseModel.model_config["arbitrary_types_allowed"] = True


def document_nodes_descriptions(output_path: str | Path):
    from .action_nodes import schemas

    behavior_library = {}

    for schema in schemas:
        node_description = (
            schema.__doc__.strip() if schema.__doc__ else "No description"
        )

        params = {}
        for field_name, field_info in schema.model_fields.items():
            # Get the full annotation string
            annotation = field_info.annotation

            if annotation is None:
                type_str = "Any"
            else:
                # str(annotation) handles List[BaseAgent], Optional[float], etc.
                # We strip "typing." to keep it clean if you prefer
                type_str = str(annotation).replace("typing.", "")

                # Clean up the common " <class '...'> " formatting for basic types
                if type_str.startswith("<class"):
                    type_str = getattr(annotation, "__name__", type_str)

            params[field_name] = {
                "type": type_str,
                "description": field_info.description or "No description provided",
            }

        behavior_library[schema.__name__] = {
            "Purpose": node_description,
            "Parameters": params,
        }

    # Save to file
    try:
        output_path = Path(output_path)
        with open(output_path, "w") as file:
            json.dump(behavior_library, file, indent=2)

        print(f"Saved behavior nodes description to {output_path.absolute()}")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    document_nodes_descriptions(
        "/opt/arena_ws/src/deps/humansim/arena_humansim/arena_humansim/behavior/behavior_nodes_library.json"
    )
