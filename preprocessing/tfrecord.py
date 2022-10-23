import tensorflow as tf



"""Function to convert a .npz dataset to a tfrecord dataset

Dont know if this is really needed , but it is a good exercise i guess """


def convert_npz_to_tfrecord(npz_path: str, tfrecord_path: str):
    """Converts a .npz dataset to a tfrecord dataset
    Args:
        npz_path (str) : path to the .npz dataset
        tfrecord_path (str) : path to save the tfrecord dataset
    """
    # Load the data
    data = load_npz_from_path(npz_path)

    # Get the data and labels
    data = data["data"]
    labels = data["labels"]

    # Create a tfrecord writer
    writer = tf.io.TFRecordWriter(tfrecord_path)

    # Loop over the data and labels
    for data, label in zip(data, labels):
        # Create a feature
        feature = {
            "data": _bytes_feature(data),
            "label": _int64_feature(label)
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    # Close the writer
    writer.close()
