"""Clearly synthetic Cortex/Delsys export for demonstrating file parsing."""


def delsys_demo_text():
    """Return one second of synthetic EMG (1 kHz) and IMU channels (100 Hz)."""
    import math

    lines = [
        "Cortex generated Asynchronous device data file (.add)",
        "[Devices]", "", "Count=1", "0, Demonstration sensor", "",
        "[Channels]", "Count=10",
        "0,0,0,EMG", "0,0,0,AccX", "0,0,0,AccY", "0,0,0,AccZ",
        "0,0,0,GyroX", "0,0,0,GyroY", "0,0,0,GyroZ",
        "0,0,0,MagX", "0,0,0,MagY", "0,0,0,MagZ", "",
        "Demonstration sensor", "EMG", "Sample,SubFrame,CortexTime,EMG",
    ]
    for sample in range(1000):
        t = sample / 1000
        emg = 0.001 * math.sin(2 * math.pi * 80 * t) * (0.3 + 0.7 * math.sin(math.pi * t) ** 2)
        lines.append(f"{sample},0,{150*t:.8f},{emg:.10f}")
    lines.extend(["", "Sample,SubFrame,CortexTime,AccX,AccY,AccZ,GyroX,GyroY,GyroZ,MagX,MagY,MagZ"])
    for sample in range(100):
        t = sample / 100
        channels = [math.sin(2 * math.pi * t), 0, 1, 0, 0, 0, 1, 0, 0]
        lines.append(f"{sample},0,{150*t:.8f}," + ",".join(map(str, channels)))
    lines.extend(["", "End"])
    return "\n".join(lines) + "\n"
