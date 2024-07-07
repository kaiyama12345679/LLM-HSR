#!/usr/bin/env python3
import sys
sys.path.append('/root/HSR/catkin_ws/src/gpsr/scripts')  # Replace with the actual path to catkin_ws/src
sys.path.append('/root/HSR/catkin_ws/src/robocup_utils/scripts/')  # Replace with the actual path to catkin_ws/src
import predefined_utils
import rospy
import utils.robot
from hsrb_interface import Robot, settings
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from llm_manager import LLMTaskPlanner, LLMWhatToDo, LLMAnswerYourSelf
from utils.control.end_effector_wrapper import GripperWrapper
from utils.control.joint_group_wrapper import JointGroupWrapper
from utils.control.mobile_base_wrapper import MobileBaseWrapper
from weblab_hsr_msgs.srv import StringTrigger, SoundPlay
from std_srvs.srv import Trigger
from world_functions import GPSRFunctions
from world_modules import GPSRModules
import json
import sys
import subprocess


class Controller:
    def __init__(self, location_file_path, predefined_file_path="/root/HSR/catkin_ws/src/robocup_utils/predefined_info/paper/locations.json"):

        # # setup.launch をバックグラウンドで実行し、ログファイルに出力をリダイレクト
        # self.setup_process = subprocess.Popen(['roslaunch', 'gpsr', 'setup.launch'],
        #                                 stdout=self.setup_log,
        #                                 stderr=subprocess.STDOUT)

        # # gpsr_modules.launch をバックグラウンドで実行し、ログファイルに出力をリダイレクト
        # self.gpsr_process = subprocess.Popen(['roslaunch', 'gpsr', 'gpsr_modules.launch'],
        #                                 stdout=self.gpsr_log,
        #                                 stderr=subprocess.STDOUT)



        #rospy.init_node('test_manager',anonymous=True)

        ##### Connect to Robot
        rospy.loginfo("Connecting to robot ..")
        rospy.set_param("/gpsr_manager/use_mic", True)
        robot = Robot()
        self.tts = robot.get('default_tts')
        rospy.loginfo("Connecting to robot 1/4")
        self.omni_base = MobileBaseWrapper(robot.get('omni_base'))
        rospy.loginfo("Connecting to robot 2/4")
        self.whole_body = JointGroupWrapper(robot.get('whole_body'))
        rospy.loginfo("Connecting to robot 3/4")
        gripper = GripperWrapper(robot.get('gripper'))
        rospy.loginfo("Connected")
        self.robot = utils.robot.Robot(robot, self.omni_base, self.whole_body, gripper)
        self.base_link_id = settings.get_frame('base')
        print("Robot Initialized.")

        # set language HSR speaks
        rospy.set_param("spoken_language", '0')  # 0: Japanese, 1: English

        self.locations = predefined_utils.load_info_from_json(location_file_path, predefined_file_path)

        with open(location_file_path, 'r') as f:
            book_data = json.load(f)

        self.books = []
        for book in book_data:
            if book["name"] != "instruction_point":
                self.books.append(book['name'])

        gpsr_modules = GPSRModules(self.robot, self.locations)
        self.gpsr_functions = GPSRFunctions(self.robot, gpsr_modules)

         # rosparam
        rospy.set_param('/scaling', True)
        rospy.loginfo("EGPSR Task Manager Initialized")

        self.neutral()
        self.speak("コントローラーの初期化が完了")

    def neutral(self):
        self.robot.whole_body.move_to_neutral()

    def speak(self, text):
        if type(text) is str:
            self.tts.say(text)
            rospy.sleep(0.4)

    def listen(self):
        is_listen_success, sentence = self.gpsr_functions.gpsr_modules.call_listen_service()
        return is_listen_success, sentence

    def move_to_with_abspos(self, x, y, yaw, **kwargs):
        self.gpsr_functions.gpsr_modules.go_to_abs_with_exception(x, y, yaw, **kwargs)
    
    def move_to_with_name(self, name):
        self.gpsr_functions.go_to_location(name)

    def release(self):
        self.robot.gripper.command(1.0)

    def pick(self):

        self.speak("本をつかみます")
        self.robot.whole_body.move_to_neutral()

        #TODO Need to tune
        tilt = -0.6
        pan = 0.0

        self.gpsr_functions.gpsr_modules.move_joints_with_exception({"head_tilt_joint": tilt, "head_pan_joint": pan})
        function_dict = {
            "function": "find_concrete_name_objects",
            "objects": "book"
        }
        # self.gpsr_functions.execute_function(function_dict, robot_at="test")

        self.gpsr_functions.find_concrete_name_objects(
                function_dict["objects"],
                function_dict.get("room"),
                function_dict.get("complement")
            )
        
        self.speak("本を見つけました")
        
        self.gpsr_functions.pick(function_dict["objects"], location_name=None)

        self.speak("本をつかみました")

if __name__ == "__main__":
    ctl = Controller("/root/HSR/catkin_ws/src/gpsr/scripts/spotting_data/kawa5.json")
    ctl.move_to_with_name("ぼっちざろっく")
    rospy.spin()