import rosbag2_py
import pandas as pd
import rclpy.serialization
from rclpy.serialization import deserialize_message
from bitbots_msgs.msg import JointCommand
import csv

# Adjust message type and topic according to your setup
MSG_TYPE = JointCommand  # Update if your message type is different
TOPIC_NAME = '/DynamixelController/command'  # Adjust the topic name accordingly

# Currently only the legs are considered, as they come together and we need no interpolation
JOINT_NAMES = [
  "LHipYaw",
  "LHipRoll",
  "LHipPitch",
  "LKnee",
  "LAnklePitch",
  "LAnkleRoll",
  "RHipYaw",
  "RHipRoll",
  "RHipPitch",
  "RKnee",
  "RAnklePitch",
  "RAnkleRoll",
]

def read_rosbag(bag_path, topic_name):
    """
    Reads joint command messages from a ROS2 bag file and returns a list of dicts containing the data.
    :param bag_path: Path to the ROS2 bag file (in MCAP format)
    :param topic_name: The name of the topic to filter the messages
    :return: List of dictionaries containing the timestamp and message data
    """
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')  # Leave empty to use default conversion

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topics and types in the bag
    topics = reader.get_all_topics_and_types()
    topic_type_map = {topic.name: topic.type for topic in topics}

    data = []

    # Check if the topic exists in the bag
    if topic_name not in topic_type_map:
        raise ValueError(f"Topic {topic_name} not found in the bag file")

    while reader.has_next():
        (topic, data_bytes, timestamp) = reader.read_next()

        # Only process messages from the target topic
        if topic == topic_name:
            msg = deserialize_message(data_bytes, MSG_TYPE)
            if set(JOINT_NAMES).issubset(set(msg.joint_names)):
                data.append({
                    'timestamp': timestamp
                } | {name: msg.positions[msg.joint_names.index(name)] for name in JOINT_NAMES})

    return data

def write_to_csv(data, output_csv):
    """
    Writes the list of joint command data to a CSV file.
    :param data: List of dictionaries containing joint command data
    :param output_csv: Path to the output CSV file
    """
    # Create a DataFrame from the data with a column for each joint + timestamp
    df = pd.DataFrame(data)
    df.to_csv(output_csv)


if __name__ == '__main__':
    # Path to the MCAP bag file
    bag_path = '/home/florian/Downloads/ID_amy_2023-07-08T12 56 52_0.mcap'  # Update with your actual bag file path
    output_csv = 'joint_commands.csv'  # Output CSV file

    # Extract joint command messages
    joint_command_data = read_rosbag(bag_path, TOPIC_NAME)

    # Write to CSV
    write_to_csv(joint_command_data, output_csv)
