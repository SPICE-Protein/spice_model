"""Data module: loading, cleaning, TFRecord, tf.data.

Note: kept lightweight here -- submodules are NOT imported eagerly to avoid a
`sys.modules` conflict when running `python -m spice_pre.data.dataset`.
Use explicit imports instead:
    from spice_pre.data.dataset import build_tfrecords, load_tfrecord_dataset
"""
