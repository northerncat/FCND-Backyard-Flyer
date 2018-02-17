import argparse
import time
from enum import Enum

import numpy as np

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection  # noqa: F401
from udacidrone.messaging import MsgID


class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5


class BackyardFlyer(Drone):

    def __init__(self, connection):
        super().__init__(connection)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.all_waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

        # define some constants for state checking
        self.TAKEOFF_ALTITUDE = 3.0
        self.TAKEOFF_CHECK_FRACTION = 0.98
        self.SQUARE_SIZE = 10.0
        self.WAYPOINT_CHECK_ERROR = 0.1
        self.LANDING_CHECK_ERROR = 0.1
        self.LANDING_CHECK_HEIGHT = 0.01

    def local_position_callback(self):
        """
        DONE: Implement this method

        This triggers when `MsgID.LOCAL_POSITION` is received and
                           self.local_position contains new data
        """
        # ignore local position if not in a mission
        if not self.in_mission:
            return
        if self.flight_state == States.TAKEOFF:
            # during takeoff phase, check whether we've reached the desired altitude
            altitude = - self.local_position[2]
            if altitude > self.TAKEOFF_CHECK_FRACTION * self.target_position[2]:
                self.all_waypoints = self.calculate_box(self.SQUARE_SIZE)
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            # during waypoint phase, check whether we've reached the target waypoint
            local_position = np.array([
                self.local_position[0],
                self.local_position[1],
                - self.local_position[2]
            ])
            distance_to_waypoint = np.linalg.norm(local_position - self.target_position)
            if distance_to_waypoint < self.WAYPOINT_CHECK_ERROR:
                # if we've reached the target waypoint, check if there are more waypoints
                if self.all_waypoints:
                    self.waypoint_transition()
                else:
                    # if no, prepare to land
                    self.landing_transition()

    def velocity_callback(self):
        """
        DONE: Implement this method

        This triggers when `MsgID.LOCAL_VELOCITY` is received and
                           self.local_velocity contains new data
        """
        # the velocity is ignored until we are ready to land
        if self.flight_state == States.LANDING:
            if ((self.global_position[2] - self.global_home[2] < self.LANDING_CHECK_ERROR) and \
                    abs(self.local_position[2]) < self.LANDING_CHECK_HEIGHT):
                self.disarming_transition()

    def state_callback(self):
        """
        DONE: Implement this method

        This triggers when `MsgID.STATE` is received and self.armed and self.guided contain new data
        """
        # pass if not in a mission
        if not self.in_mission:
            return
        if self.flight_state == States.MANUAL:
            # starts arming when in a mission but still manual mode
            self.arming_transition()
        elif self.flight_state == States.ARMING:
            if self.armed:
                # after arming, prepare to takeoff
                self.takeoff_transition()
        elif self.flight_state == States.DISARMING:
            if not self.armed:
                # after disarming, prepare to transition back to manual mode
                self.manual_transition()


    def calculate_box(self, length):
        """ DONE: Fill out this method

        1. Return waypoints to fly a box
        """
        return [
            [length, 0.0, self.TAKEOFF_ALTITUDE, 0.0],
            [length, length, self.TAKEOFF_ALTITUDE, 0.0],
            [0.0, length, self.TAKEOFF_ALTITUDE, 0.0],
            [0.0, 0.0, self.TAKEOFF_ALTITUDE, 0.0],
        ]

    def arming_transition(self):
        """ DONE: Fill out this method

        1. Take control of the drone
        2. Pass an arming command
        3. Set the home location to current position
        4. Transition to the ARMING state
        """
        print("arming transition")
        self.take_control()
        self.arm()
        # set the current location to be the home position
        self.set_home_position(self.global_position[0],
                               self.global_position[1],
                               self.global_position[2])
        self.flight_state = States.ARMING

    def takeoff_transition(self):
        """ DONE: Fill out this method

        1. Set target_position altitude to 3.0m
        2. Command a takeoff to 3.0m
        3. Transition to the TAKEOFF state
        """
        print("takeoff transition")
        self.target_position[2] = self.TAKEOFF_ALTITUDE
        self.takeoff(self.TAKEOFF_ALTITUDE)
        self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        """ DONE: Fill out this method

        1. Command the next waypoint position
        2. Transition to WAYPOINT state
        """
        print("waypoint transition")
        target = self.all_waypoints.pop(0)
        self.target_position = np.array(target[0:3])
        self.cmd_position(*target)
        self.flight_state = States.WAYPOINT

    def landing_transition(self):
        """ DONE: Fill out this method

        1. Command the drone to land
        2. Transition to the LANDING state
        """
        print("landing transition")
        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        """ DONE: Fill out this method

        1. Command the drone to disarm
        2. Transition to the DISARMING state
        """
        print("disarm transition")
        self.disarm()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        """This method is provided

        1. Release control of the drone
        2. Stop the connection (and telemetry log)
        3. End the mission
        4. Transition to the MANUAL state
        """
        print("manual transition")
        self.release_control()
        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def start(self):
        """This method is provided

        1. Open a log file
        2. Start the drone connection
        3. Close the log file
        """
        print("Creating log file")
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()
        print("Closing log file")
        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help="host address, i.e. '127.0.0.1'"
    )
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False)
    #conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
    drone = BackyardFlyer(conn)
    time.sleep(2)
    drone.start()
