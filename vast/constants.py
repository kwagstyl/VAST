
import os

for civet_path in ['/data1/quarantines/Linux-x86_64/bin/']:
    if os.path.exists(civet_path):
        print('Setting civet_path to ' + civet_path)
        break
if not os.path.exists(civet_path):
    print('WARNING: civet_path not found, setting to None.')
    print('Some functions depend on CIVET, consider installing.')
    civet_path = None


    