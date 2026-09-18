#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace snd::trident;
#pragma link C++ defined_in namespace snd::trident;

#pragma link C++ enum snd::trident::ProcessType;
#pragma link C++ enum snd::trident::RegionType;
#pragma link C++ enum snd::trident::FlagBits;

#pragma link C++ struct snd::trident::TridentTruthInfo+;
#pragma link C++ struct snd::trident::TridentTruthConfig+;
#pragma link C++ struct snd::trident::AncestryResult+;
#pragma link C++ struct snd::trident::FiducialResult+;
#pragma link C++ class snd::trident::TridentTruthProcessor+;
#pragma link C++ struct snd::trident::PreselectionMetrics+;
#pragma link C++ class snd::trident::PreselectionProcessor+;
#pragma link C++ struct snd::trident::IP1Filter+;
#pragma link C++ struct snd::trident::SciFiAnisotropyCalculator+;
#pragma link C++ struct snd::trident::MuFilterAnisotropyCalculator+;
#pragma link C++ struct snd::trident::MuonCalibrationConfig+;
#pragma link C++ struct snd::trident::MuonCalibrationMetrics+;
#pragma link C++ class snd::trident::MuonCalibrationProcessor+;
#pragma link C++ enum snd::trident::MuonTruthCategory;
#pragma link C++ struct snd::trident::PassingMuonTruthConfig+;
#pragma link C++ struct snd::trident::PassingMuonTruthInfo+;
#pragma link C++ class snd::trident::PassingMuonTruthProcessor+;
#pragma link C++ struct snd::trident::ChannelGeometry+;
#pragma link C++ struct snd::trident::DigiValidationSummary+;
#pragma link C++ class snd::trident::DigiValidationProcessor+;
#pragma link C++ class snd::trident::ProgressPrinter+;

#pragma link C++ namespace snd;
#pragma link C++ class snd::DataManager+;

#endif
