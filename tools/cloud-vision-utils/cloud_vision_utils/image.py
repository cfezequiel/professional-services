# python3
"""Batch image processing."""

import io
import logging
import os

import piexif
from PIL import Image
from tensorflow.io import gfile

ROTATE_VALUES = {3: 180, 6: 270, 8: 90}


def save(filename: str, image: Image, exif: str):
  with gfile.GFile(filename, 'wb') as f:
    image.save(f, exif=exif)


# pylint: disable=protected-access
def autorotate(filename: str):
  """Rotate PIL Image based on EXIF orientation."""

  with gfile.GFile(filename, 'rb') as f:
    img = Image.open(io.BytesIO(f.read()))

  if 'exif' in img.info:
    exif_dict = piexif.load(img.info['exif'])
    if piexif.ImageIFD.Orientation in exif_dict.get('0th'):
      orientation = exif_dict['0th'].pop(piexif.ImageIFD.Orientation)
      exif_bytes = piexif.dump(exif_dict)

      def _rotate(image, angle):
        return image.rotate(angle, resample=Image.BICUBIC, expand=True)

      if orientation == 2:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
      elif orientation == 3:
        img = _rotate(img, 180)
      elif orientation == 4:
        img = _rotate(img, 180).transpose(Image.FLIP_LEFT_RIGHT)
      elif orientation == 5:
        img = _rotate(img, -90).transpose(Image.FLIP_LEFT_RIGHT)
      elif orientation == 6:
        img = _rotate(img, -90)
      elif orientation == 7:
        img = _rotate(img, 90).transpose(Image.FLIP_LEFT_RIGHT)
      elif orientation == 8:
        img = _rotate(img, 90)

      save(filename, img, exif=exif_bytes)


def autorotate_batch(input_dir: str):
  """Autorotate images in a nested directory structure."""

  for topdir, _, files in gfile.walk(input_dir):
    for f in files:
      logging.info('Processing %s', f)
      filename = os.path.join(topdir, f)
      try:
        autorotate(filename)
      except Exception as e:
        logging.warning('autorotate failed for {}. Reason: {}'
                        .format(f, str(e)))
