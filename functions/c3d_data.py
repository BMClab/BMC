"""Read C3D data with the pure-Python c3d package on desktop and browser runtimes."""


def load_c3d(path):
    """Return metadata, marker coordinates, and calibrated analog samples.

    Times are seconds relative to the first recorded point frame. Negative
    residuals mark invalid points; a zero coordinate is a valid measurement.
    Analog scaling is performed once by c3d.Reader. Analog channels are not
    assumed to be force components or rotated in arbitrary groups of three.
    """
    import c3d
    import numpy as np
    import pandas as pd

    def normalized_labels(reader, group, used):
        # USED defines data width; LABELS can legally be shorter or longer.
        labels = []
        suffix = ""
        while len(labels) < used:
            parameter = reader.get(f"{group}:LABELS{suffix}")
            if parameter is None:
                break
            labels.extend(parameter.string_array)
            suffix = 2 if suffix == "" else suffix + 1
        labels = [str(label).strip(" \t\r\n\x00") for label in labels[:used]]
        labels.extend([""] * (used - len(labels)))
        reserved = set(labels)
        seen = set()
        result = []
        for index, label in enumerate(labels):
            base = label or f"{group}_{index + 1}"
            candidate = base
            number = 2
            while candidate in seen or (candidate != label and candidate in reserved):
                candidate = f"{base}_{number}"
                number += 1
            result.append(candidate)
            seen.add(candidate)
        return result

    with open(path, "rb") as handle:
        reader = c3d.Reader(handle)
        frames = list(reader.read_frames())
        if not frames:
            raise ValueError("The C3D file contains no data frames")
        point_labels = normalized_labels(reader, "POINT", reader.point_used)
        analog_labels = normalized_labels(reader, "ANALOG", reader.analog_used)
        units = reader.get("POINT:UNITS")
        event_used = reader.get("EVENT:USED")
        events = reader.get("EVENT:LABELS")
        event_times = reader.get("EVENT:TIMES")
        info = {
            "point_rate": float(reader.point_rate),
            "analog_rate": float(reader.analog_rate),
            "first_frame": int(reader.first_frame),
            "frames": len(frames),
            "point_units": units.string_value.strip().strip("\x00")
            if units
            else "unspecified",
            "analog_units": list(reader.get("ANALOG:UNITS").string_array)
            if reader.get("ANALOG:UNITS")
            else [],
            "parameter_groups": {
                name: [key for key, _ in group.param_items()]
                for name, group in reader.group_items()
            },
            "events": list(events.string_array) if events else [],
            "event_times": event_times.float_array.tolist() if event_times else [],
            "event_count": int(event_used.int16_value) if event_used else 0,
        }
        all_points = np.stack([points for _, points, _ in frames])
        xyz = all_points[:, :, :3].copy()
        xyz[all_points[:, :, 3] < 0] = np.nan
        columns = pd.MultiIndex.from_product(
            [point_labels, list("XYZ")], names=["Marker", "Coordinate"]
        )
        points = pd.DataFrame(
            xyz.reshape(len(frames), -1),
            columns=columns,
            index=np.arange(len(frames)) / reader.point_rate,
        )
        if reader.analog_used:
            samples = np.concatenate([analog for _, _, analog in frames], axis=1).T
            analog = pd.DataFrame(
                samples,
                columns=analog_labels,
                index=np.arange(len(samples)) / reader.analog_rate,
            )
        else:
            analog = pd.DataFrame()
    points.index.name = analog.index.name = "Time [s]"
    return info, points, analog
