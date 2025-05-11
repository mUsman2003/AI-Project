import msgParser
import carState
import carControl
import keyboard
import csv
import time
from datetime import datetime
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

class Driver(object):
    """
    Driver class for SCRC: interfaces with the AI model and sends control commands
    """

    def __init__(self, stage, ai_mode=True):  # AI mode enabled by default
        """Initialize driver parameters and load AI model"""
        # Define race stages
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.ai_mode = ai_mode

        # Initialize parser, state tracker, and control interface
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        # Control constraints
        self.steer_lock = 1.0472  # max steering angle (rad)
        self.max_speed = 450      # km/h
        self.prev_rpm = None      # for gear shifting logic

        # Set initial gear and neutral controls
        self.control.setGear(1)
        self.control.setAccel(0.0)
        self.control.setBrake(0.0)
        self.control.setSteer(0.0)

        # Load AI model and scalers
        self._load_ai_model()

        # Prepare logging file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f"ai_car_data_{timestamp}.csv"
        self._initialize_logger()

    def _load_ai_model(self):
        """Attempt to load pre-trained neural network and scalers"""
        try:
            with open('trained_NN.pkl', 'rb') as f:
                package = pickle.load(f)

            self.model = package['model']
            self.scaler_X = package['scaler_X']
            self.scaler_y_cont = package['scaler_y_cont']
            self.input_features = package['input_features']
            self.continuous_outputs = package['continuous_outputs']
            self.binary_outputs = package['binary_outputs']
            self.steer_peaks = package['steer_peaks']

            print("[INFO] Loaded AI model and preprocessing tools successfully.")
        except Exception as err:
            print(f"[WARNING] Failed to load AI model: {err}")
            self.ai_mode = False
            print("[INFO] Switching to fallback control mode.")

    def _initialize_logger(self):
        """Create CSV file and write header for telemetry data"""
        header = [
            'Timestamp', 'Accel', 'Brake', 'Gear', 'Steer', 'Clutch', 'Focus', 'Meta',
            'Angle', 'LapTime', 'Damage', 'DistStart', 'DistRaced',
            'Fuel', 'GearState', 'LastLap', 'RacePos', 'RPM', 'SpeedX', 'SpeedY', 'SpeedZ',
            'TrackPos', 'Z', 'AI_Mode'
        ]
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

    def prepare_input_data(self):
        """Extract features from current state for AI prediction"""
        data = {}
        # Basic scalar features
        data['Angle'] = self.state.getAngle() or 0
        data['CurLapTime'] = self.state.getCurLapTime() or 0
        data['Damage'] = self.state.getDamage() or 0
        data['DistFromStart'] = self.state.getDistFromStart() or 0
        data['DistRaced'] = self.state.getDistRaced() or 0
        data['Fuel'] = self.state.getFuel() or 0
        data['LastLapTime'] = getattr(self.state, 'lastLapTime', 0) or 0
        data['RPM'] = self.state.getRpm() or 0
        data['Speed X'] = self.state.getSpeedX() or 0
        data['Speed Y'] = self.state.getSpeedY() or 0
        data['Speed Z'] = self.state.getSpeedZ() or 0
        data['TrackPos'] = self.state.getTrackPos() or 0
        data['RacePos'] = self.state.getRacePos() or 0
        data['Gear2'] = self.state.getGear() or 1

        # Wheel spin velocities
        wheels = self.state.getWheelSpinVel() or []
        for i in range(4):
            data[f'WheelSpinVel_{i}'] = wheels[i] if i < len(wheels) else 0

        # Track range sensors
        track = self.state.getTrack() or []
        for i in range(19):
            data[f'Track_{i}'] = track[i] if i < len(track) else 200

        # Focus sensors
        focus = getattr(self.state, 'focus', [])
        for i in range(5):
            data[f'Focus2_{i}'] = focus[i] if i < len(focus) else 0

        # Placeholder for derived features
        data['Steer_Diff'] = 0
        data['Steer_Accel'] = 0
        speed_x = abs(data['Speed X'])
        data['Angle_Speed_Ratio'] = data['Angle'] / (speed_x if speed_x > 0.1 else 1)

        return data

    def predict_controls(self, input_data):
        """Generate control commands via the AI model"""
        # Build input array ordered by feature list
        arr = np.zeros((1, len(self.input_features)))
        for idx, feat in enumerate(self.input_features):
            arr[0, idx] = input_data.get(feat, 0)

        scaled = self.scaler_X.transform(arr)
        preds = self.model.predict(scaled)

        commands = {}
        # Continuous outputs
        if self.continuous_outputs:
            cont_idx = preds if not isinstance(preds, list) else preds[0]
            unscaled = self.scaler_y_cont.inverse_transform(cont_idx.reshape(1, -1))[0]
            for i, feat in enumerate(self.continuous_outputs):
                val = unscaled[i]
                if feat == 'Steer':
                    for peak in self.steer_peaks:
                        if abs(val - peak) < 0.05:
                            val = peak
                            break
                commands[feat] = val

        # Binary outputs follow continuous
        offset = len(self.continuous_outputs) or 0
        for i, feat in enumerate(self.binary_outputs):
            raw = preds[offset + i] if isinstance(preds, list) else preds[:, i]
            commands[feat] = int(raw[0] > 0.5)

        return commands

    def drive(self, msg):
        """Process incoming message, choose control strategy, and log data"""
        self.state.setFromMsg(msg)

        if self.ai_mode:
            try:
                data = self.prepare_input_data()
                controls = self.predict_controls(data)

                # Set outputs if available
                if 'Acceleration' in controls:
                    self.control.setAccel(float(controls['Acceleration']))
                if 'Brake' in controls:
                    self.control.setBrake(float(controls['Brake']))
                if 'Gear' in controls:
                    self.control.setGear(int(controls['Gear']))
                if 'Steer' in controls:
                    self.control.setSteer(float(controls['Steer']))
                if 'Clutch' in controls:
                    self.control.setClutch(float(controls['Clutch']))

                self._gear_logic()
                self._speed_adjustment()
            except Exception as err:
                print(f"[ERROR] AI drive failed: {err}")
                self.basic_ai_control()
        else:
            self.basic_ai_control()

        self._write_log()
        return self.control.toMsg()

    def init(self):
        """Generate initialization string for range sensors"""
        angles = [0] * 19
        # wide angles first
        for i in range(5):
            angles[i] = -90 + 15 * i
            angles[18 - i] = 90 - 15 * i
        # narrow angles next
        for i in range(5, 9):
            angles[i] = -20 + 5 * (i - 5)
            angles[18 - i] = 20 - 5 * (i - 5)
        return self.parser.stringify({'init': angles})

    def _gear_logic(self):
        """Adjust gear based on RPM and speed"""
        rpm = self.state.getRpm()
        gear = self.control.getGear()
        speed = abs(self.state.getSpeedX())

        if gear == -1 or rpm is None:
            return
        if rpm > 7000 and gear < 6:
            self.control.setGear(gear + 1)
        elif rpm < 3000 and gear > 1:
            self.control.setGear(gear - 1)
        if speed < 10 and gear > 1:
            self.control.setGear(1)

    def _speed_adjustment(self):
        """Modulate throttle and brake based on track sensors"""
        track = self.state.getTrack()
        speed = abs(self.state.getSpeedX())
        if not track or len(track) < 19:
            return

        # side sensor distances
        left = track[9]
        right = track[18]
        if min(left, right) < 5:
            self.control.setAccel(max(0, self.control.getAccel() - 0.2))
            self.control.setBrake(min(1, self.control.getBrake() + 0.1))

        # front curvature check
        front = track[5:14]
        if min(front) < 20 and speed > 50:
            self.control.setAccel(max(0, self.control.getAccel() - 0.3))
            self.control.setBrake(min(1, self.control.getBrake() + 0.1))

    def basic_ai_control(self):
        """Default driving behavior if AI fails"""
        self.steer()
        self.gear()
        self.speed()

    def steer(self):
        """Compute basic steering based on angle and track position"""
        ang = self.state.angle
        pos = self.state.trackPos
        val = (ang * 1.2 - pos * 0.5) * 0.5
        # rate limiter
        curr = self.control.getSteer() or 0
        delta = 0.25
        val = max(min(curr + delta, val), curr - delta)
        self.control.setSteer(val)

    def gear(self):
        """Simple gear shifting logic"""
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        if gear == -1:
            return
        up = True if self.prev_rpm is None else (self.prev_rpm < rpm)
        if up and rpm > 7000 and gear < 6:
            gear += 1
        elif not up and rpm < 3000 and gear > 1:
            gear -= 1
        gear = max(1, min(6, gear))
        self.control.setGear(gear)
        self.prev_rpm = rpm

    def speed(self):
        """Manage throttle/brake to maintain speed target"""
        speed = abs(self.state.getSpeedX())
        gear = self.state.getGear()
        if gear == -1:
            if speed < 30:
                self.control.setAccel(0.5)
                self.control.setBrake(0.0)
            else:
                self.control.setAccel(0.0)
                self.control.setBrake(0.1)
        else:
            if speed < self.max_speed:
                self.control.setAccel(1.0)
                self.control.setBrake(0.0)
            else:
                self.control.setAccel(0.0)
                self.control.setBrake(0.1)

    def _write_log(self):
        """Append current controls and state to CSV"""
        try:
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    self.control.getAccel() or 0,
                    self.control.getBrake() or 0,
                    self.control.getGear() or 1,
                    self.control.getSteer() or 0,
                    self.control.getClutch() or 0,
                    self.control.focus or 0,
                    self.control.getMeta() or 0,
                    self.state.getAngle() or 0,
                    self.state.getCurLapTime() or 0,
                    self.state.getDamage() or 0,
                    self.state.getDistFromStart() or 0,
                    self.state.getDistRaced() or 0,
                    self.state.getFuel() or 0,
                    self.state.getGear() or 0,
                    getattr(self.state, 'lastLapTime', 0) or 0,
                    self.state.getRacePos() or 0,
                    self.state.getRpm() or 0,
                    self.state.getSpeedX() or 0,
                    self.state.getSpeedY() or 0,
                    self.state.getSpeedZ() or 0,
                    self.state.getTrackPos() or 0,
                    self.state.getZ() or 0,
                    self.ai_mode
                ])
        except Exception as err:
            print(f"[ERROR] Failed to log data: {err}")

    def onShutDown(self):
        """Cleanup on shutdown event"""
        keyboard.unhook_all()
        print("[INFO] Driver has been terminated.")

    def onRestart(self):
        """Reset controls on restart event"""
        self.control.setGear(1)
        self.control.setAccel(0.0)
        self.control.setBrake(0.0)
        self.control.setSteer(0.0)
        print("[INFO] Driver controls have been reset.")
