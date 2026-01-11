import time
import serial
import numpy as np
from collections import deque
from datetime import datetime

# ---------------------------------------------------------------------------
# Simple logging helper
# Prints a timestamped message to stdout so we can see what the simulator is
# doing in real time.
# ---------------------------------------------------------------------------
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# LaserPlant
# This class simulates the *physical behaviour* of a laser power control system.
# It is NOT a real laser model, but is realistic enough to:
#   - test serial communication
#   - test PID behaviour
#   - test optimisation logic
#
# The model used is:
#   - first-order system (typical for thermal / power systems)
#   - optional dead-time (transport delay)
#   - noise (sensor noise)
#   - saturation limits (safety constraints)
# ---------------------------------------------------------------------------
class LaserPlant:
    def __init__(
        self,
        tau=0.35,          # Time constant of the plant (how fast it reacts)
        dead_time=0.08,    # Transport delay before control has an effect
        dt=0.01,           # Simulation timestep (seconds)
        noise_std=0.003,   # Measurement noise
        u_min=0.0,         # Minimum control effort
        u_max=1.0          # Maximum control effort
    ):
        # Store parameters
        self.tau = float(tau)
        self.dead_time = float(dead_time)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        # Internal plant state
        self.y = 0.0       # Measured output (laser power)
        self.u = 0.0       # Control input
        self.sp = 0.0      # Setpoint

        # Implement dead-time using a FIFO queue
        # Control commands are delayed before affecting the plant
        self.delay_steps = max(0, int(round(self.dead_time / self.dt)))
        self.u_queue = deque(
            [0.0] * (self.delay_steps + 1),
            maxlen=self.delay_steps + 1
        )

    # -----------------------------------------------------------------------
    # step()
    # Advances the simulated plant by one timestep.
    # Called repeatedly during a "START" trial.
    # -----------------------------------------------------------------------
    def step(self, u_cmd):
        # Enforce actuator limits (safety constraint)
        u_cmd = float(np.clip(u_cmd, self.u_min, self.u_max))
        self.u = u_cmd

        # Push command into delay line
        self.u_queue.append(u_cmd)
        u_delayed = self.u_queue[0]

        # First-order plant dynamics:
        #   dy/dt = -(y - u_delayed) / tau
        dy = (-(self.y - u_delayed) / self.tau) * self.dt
        self.y += dy

        # Add simulated sensor noise
        self.y += np.random.normal(0.0, self.noise_std)

        # Clamp measurement to plausible bounds
        self.y = float(np.clip(self.y, -0.05, 1.05))

        return self.y, u_delayed


# ---------------------------------------------------------------------------
# PID Controller
# Standard discrete-time PID controller with:
#   - proportional term
#   - integral term
#   - derivative term
#   - simple anti-windup via integral rollback on saturation
# ---------------------------------------------------------------------------
class PID:
    def __init__(
        self,
        kp=0.2,
        ki=0.1,
        kd=0.0,
        dt=0.01,
        u_min=0.0,
        u_max=1.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max

        self.integral = 0.0
        self.prev_err = 0.0

    # Reset controller state (called at start of each trial)
    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0

    # Compute next control output
    def update(self, sp, y):
        err = sp - y

        # Integral term
        self.integral += err * self.dt

        # Derivative term
        derr = (err - self.prev_err) / self.dt
        self.prev_err = err

        # PID equation
        u = (
            self.kp * err +
            self.ki * self.integral +
            self.kd * derr
        )

        # Saturate output
        u_sat = float(np.clip(u, self.u_min, self.u_max))

        # Anti-windup: undo last integral step if saturated
        if u != u_sat:
            self.integral -= err * self.dt

        return u_sat


# ---------------------------------------------------------------------------
# main()
# Entry point of the simulator.
# Opens a serial port and waits for commands from a controller.
# ---------------------------------------------------------------------------
def main():
    import argparse

    # Command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--port",
        required=True,
        help="Serial port (e.g. /dev/ttyUSB0)"
    )
    ap.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate"
    )
    args = ap.parse_args()

    log("Starting Laser Simulator")
    log(f"Opening serial port {args.port} @ {args.baud} baud")

    # Open serial port
    ser = serial.Serial(args.port, args.baud, timeout=0.1)

    # Notify controller that simulator is alive
    ser.write(b"OK SIM READY\n")
    log("Serial port opened, simulator ready")

    # Create plant and PID controller
    plant = LaserPlant()
    pid = PID(dt=plant.dt)

    # Helper to send a line over serial
    def send(line):
        ser.write((line + "\n").encode("ascii", errors="ignore"))

    buffer = b""

    log("Waiting for commands...")

    # -----------------------------------------------------------------------
    # Main serial receive loop
    # -----------------------------------------------------------------------
    while True:
        chunk = ser.read(256)

        # No data received → continue waiting
        if not chunk:
            continue

        buffer += chunk

        # Process complete lines
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            cmd = raw.decode("ascii", errors="ignore").strip()

            if not cmd:
                continue

            log(f"RX: {cmd}")

            parts = cmd.split()
            head = parts[0].upper()

            # ---------------------------------------------------------------
            # PING command — connectivity test
            # ---------------------------------------------------------------
            if head == "PING":
                send("OK PONG")
                log("TX: OK PONG")

            # ---------------------------------------------------------------
            # SET SP <value> — update setpoint
            # ---------------------------------------------------------------
            elif head == "SET" and parts[1].upper() == "SP":
                plant.sp = float(parts[2])
                send(f"OK SP {plant.sp:.4f}")
                log(f"Setpoint set to {plant.sp:.4f}")

            # ---------------------------------------------------------------
            # SET PID <kp> <ki> <kd> — update PID gains
            # ---------------------------------------------------------------
            elif head == "SET" and parts[1].upper() == "PID":
                pid.kp = float(parts[2])
                pid.ki = float(parts[3])
                pid.kd = float(parts[4])
                pid.reset()
                send(f"OK PID {pid.kp:.4f} {pid.ki:.4f} {pid.kd:.4f}")
                log(f"PID updated: kp={pid.kp}, ki={pid.ki}, kd={pid.kd}")

            # ---------------------------------------------------------------
            # START <seconds> — run a closed-loop trial
            # ---------------------------------------------------------------
            elif head == "START":
                seconds = float(parts[1])
                steps = int(seconds / plant.dt)
                pid.reset()

                log(f"Starting trial for {seconds:.2f}s ({steps} steps)")

                aborted = False

                for i in range(steps):
                    # Compute control output
                    u_cmd = pid.update(plant.sp, plant.y)

                    # Advance plant
                    y, _ = plant.step(u_cmd)

                    status = "OK"

                    # Safety abort condition
                    if y > 1.03:
                        status = "ABORT"
                        aborted = True

                    # Send telemetry
                    send(
                        f"DATA t={i*plant.dt:.4f} "
                        f"y={y:.6f} sp={plant.sp:.6f} "
                        f"u={u_cmd:.6f} status={status}"
                    )

                    if aborted:
                        log("ABORT triggered — stopping trial")
                        break

                send(f"OK DONE aborted={int(aborted)}")
                log("Trial complete")

            # ---------------------------------------------------------------
            # Unknown command
            # ---------------------------------------------------------------
            else:
                send("ERR UNKNOWN_CMD")
                log("Unknown command received")


# ---------------------------------------------------------------------------
# Program entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
