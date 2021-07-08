./ActsExampleParticleGun -n100 --gen-nparticles 1 --gen-mom-gev 1.:1. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
./ActsExampleFatrasDD4hep --dd4hep-input=../../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv  --bf-constant-tesla 0:0:2 --input-dir="./"
