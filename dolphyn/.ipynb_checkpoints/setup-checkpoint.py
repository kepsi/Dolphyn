#!/usr/bin/env python
# coding: utf-8

import setuptools

setuptools.setup(name='dolphyn',
version='0.1',
description='Design package for efficient PhIP-Seq libraries',
author='Anna Liebhoff',
author_email='aliebho1@jhu.edu',
license='MIT',
packages=setuptools.find_packages(),
py_modules=['dolphyn'],
include_package_data=True,
package_data={'': ['trainingdata/*.csv']},
zip_safe=False)