@REM ---------------------------------------------------   pdg-11   ---------------------------------------------------

@REM First Dataset: Ideal dataset
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 11 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 11 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --bf-constant-tesla 0:0:2 --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg11/pdg11-n25-0.5to10GeV-0.5eta/

@REM Second Dataset: With Material Effects (and constant B)
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 11 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 11 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --mat-input-type=file --mat-input-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-material-map.root --bf-constant-tesla 0:0:2 --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg11/pdg11-n25-0.5to10GeV-0.5eta-with-material-effects/

@REM Third Dataset: With Non Homogenous B (without Material Effects)
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 11 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 11 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --bf-map-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-bfield.root --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg11/pdg11-n25-0.5to10GeV-0.5eta-non-homogenous-magnetic-field/

@REM Fourth Dataset: With Material Effects and Non Homogenous B
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 11 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 11 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --mat-input-type=file --mat-input-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-material-map.root --bf-map-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-bfield.root --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg11/pdg11-n25-0.5to10GeV-0.5eta-with-material-effects-non-homogenous-magnetic-field/

@ REM -----------------------------------------------------------------------------------------------------------------




@REM ---------------------------------------------------   pdg-13   ---------------------------------------------------

@REM First Dataset: With Material Effects (and constant B)
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 13 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 13 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --mat-input-type=file --mat-input-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-material-map.root --bf-constant-tesla 0:0:2 --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg13/pdg13-n25-0.5to10GeV-0.5eta-with-material-effects/

@REM Second Dataset: With Non Homogenous B (without Material Effects)
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 13 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 13 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --bf-map-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-bfield.root --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg13/pdg13-n25-0.5to10GeV-0.5eta-non-homogenous-magnetic-field/

@REM Third Dataset: With Material Effects and Non Homogenous B
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 13 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 13 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --mat-input-type=file --mat-input-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-material-map.root --bf-map-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-bfield.root --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg13/pdg13-n25-0.5to10GeV-0.5eta-with-material-effects-non-homogenous-magnetic-field/

@ REM -----------------------------------------------------------------------------------------------------------------




@REM ---------------------------------------------------   pdg-211   ---------------------------------------------------

@REM First Dataset: Ideal dataset
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 211 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 211 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --bf-constant-tesla 0:0:2 --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg211/pdg211-n25-0.5to10GeV-0.5eta/

@REM Second Dataset: With Material Effects (and constant B)
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 211 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 211 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --mat-input-type=file --mat-input-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-material-map.root --bf-constant-tesla 0:0:2 --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg211/pdg211-n25-0.5to10GeV-0.5eta-with-material-effects/

@REM Third Dataset: With Non Homogenous B (without Material Effects)
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 211 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 211 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --bf-map-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-bfield.root --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg211/pdg211-n25-0.5to10GeV-0.5eta-non-homogenous-magnetic-field/

@REM Fourth Dataset: With Material Effects and Non Homogenous B
../acts_build/bin/ActsExampleParticleGun -n 100 --gen-pdg 211 --gen-nparticles 25 --gen-mom-gev 0.5:10. --gen-mom-transverse true --gen-eta -0.5:0.5 --output-csv
../acts_build/bin/ActsExampleFatrasDD4hep --gen-pdg 211 --dd4hep-input=../acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml --output-csv --mat-input-type=file --mat-input-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-material-map.root --bf-map-file=../../../Desktop/CERN/other-repos/OpenDataDetector/data/odd-bfield.root --input-dir="./"
mv ./* ../../../Desktop/CERN/tra-tra/data/pdg211/pdg211-n25-0.5to10GeV-0.5eta-with-material-effects-non-homogenous-magnetic-field/

@ REM -----------------------------------------------------------------------------------------------------------------
