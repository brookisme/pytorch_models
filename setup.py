from distutils.core import setup
setup(
  name = 'pytorch_models',
  packages = ['pytorch_models'],
  version = '0.0.0.1',
  description = 'PyTorch Models',
  author = 'Brookie Guzder-Williams',
  author_email = 'brook.williams@gmail.com',
  url = 'https://github.com/brookisme/pytorch_models',
  download_url = 'https://github.com/brookisme/pytorch_models/tarball/0.1',
  keywords = ['PyTorch','CNN','Neural Networks','Machine learning','Deep learning'],
  include_package_data=True,
  data_files=[
    (
      'config',[]
    )
  ],
  classifiers = [],
  entry_points={
      'console_scripts': [
      ]
  }
)