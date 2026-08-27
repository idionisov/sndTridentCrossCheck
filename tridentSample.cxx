#include <iostream>
#include <vector>

#include "TObject.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TClonesArray.h"
#include "TH1D.h"
#include "TError.h"

#include "ShipMCTrack.h"

// Cuts
#include "sndBaseCut.h"
#include "sndMinSciFiHitsCut.h"
#include "sndSciFiStationCut.h"
#include "sndVetoCut.h"
#include "sndMinSciFiConsecutivePlanes.h"
#include "sndDSActivityCut.h"
#include "sndUSQDCCut.h"
#include "sndEventDeltat.h"
#include "sndAvgSciFiFiducialCut.h"
#include "sndAvgDSFiducialCut.h"
#include "sndDSVetoCut.h"
#include "sndHasVetoHitsCut.h"
#include "sndMinSciFiPlanesCut.h"
#include "sndTridentHitsCut.h"
#include "sndMaxSciFiHitsCut.h"
#include "sndMaxPlaneSciFiHitsCut.h"
#include "sndMaxSciFiSignalCut.h"
#include "sndMaxPlaneSciFiSignalCut.h"
#include "sndTridentDensityCut.h"
#include "sndDSQDCCut.h"
#include "sndMinDSHitsCut.h"
#include "sndLastDSPlaneCut.h"
#include "sndMinUSHitsCut.h"
#include "sndMinMaxSciFiSignalCut.h"
#include "sndMinMaxUSQDCCut.h"
#include "sndDStoSciFiQDCRatioCut.h"

// Alternative sets of cuts.
enum Cutset { stage1cuts, novetocuts, FVsideband, allowWalls2and5, stage1cutsVetoFirst, nueFilter, tridentSelection, looseTridentCuts, rockTridentPreselection} ;

int main(int argc, char ** argv) {
  gErrorIgnoreLevel = kWarning;

  std::cout << "Starting trident filter" << std::endl;

  if (argc != 4) {
    std::cout << "Three arguments required: input file name (or reg exp), output file name, cut set (0: stage 1 selection, 1: no veto or scifi 1st layer, 2: FV sideband, 3: include walls 2 and 5, 4: stage1cutsVetoFirst, 5: nue filter, 6: trident selection (with veto hits), 7: loose trident cuts, 8: rock trident preselection)." << std::endl;
    return -1;
  }

  // Input files
  bool isMC = false;
  std::string tree_name = "rawConv";

  // Cleanly probe tree name without noisy TChain error messages
  TFile * test_file = TFile::Open(argv[1]);
  if (test_file && !test_file->IsZombie()) {
    TTree * t = (TTree*) test_file->Get("cbmsim");
    if (!t) t = (TTree*) test_file->Get("rawConv");
    if (t) {
      tree_name = t->GetName();
      isMC = (t->GetBranch("MCTrack") != nullptr);
    }
    test_file->Close();
    delete test_file;
  }

  TChain * ch = new TChain(tree_name.c_str());
  ch->Add(argv[1]);
  if (ch->GetEntries() == 0) {
    // Fallback attempt
    delete ch;
    tree_name = (tree_name == "rawConv") ? "cbmsim" : "rawConv";
    ch = new TChain(tree_name.c_str());
    ch->Add(argv[1]);
    if (ch->GetEntries() > 0) {
      isMC = (ch->GetBranch("MCTrack") != nullptr);
    } else {
      std::cout << "Didn't find rawConv or cbmsim in input file" << std::endl;
      exit(-1);
    }
  } else {
    isMC = (ch->GetBranch("MCTrack") != nullptr);
  }

  if (isMC) {
    if (ch->GetBranch("EventHeader")) ch->SetBranchStatus("EventHeader", 0);
    if (ch->GetBranch("EventHeader.")) ch->SetBranchStatus("EventHeader.", 0);
  }
  std::cout << "Got input tree (" << tree_name << ", " << (isMC ? "MC" : "Data") << ")" << std::endl;

  // MC truth
  TClonesArray * MCTracks = new TClonesArray("ShipMCTrack", 5000);
  Double_t mc_weight = 1.0;
  bool has_mc_weight = false;
  Bool_t is_signal = false;
  Double_t inv_mass_2mu = 0.0;
  Double_t opening_angle_mrad = 0.0;
  Double_t p_mu_in = 0.0;
  Double_t p_mu_minus = 0.0;
  Double_t p_mu_plus = 0.0;
  Double_t vtx_x = 0.0;
  Double_t vtx_y = 0.0;
  Double_t vtx_z = -9999.0;
  Double_t energy_asym = 0.0;
  Double_t pt_2mu = 0.0;
  bool has_trident_truth = false;

  if (isMC) {
    ch->SetBranchAddress("MCTrack", &MCTracks);
    if (ch->GetBranch("mc_weight")) {
      ch->SetBranchAddress("mc_weight", &mc_weight);
      has_mc_weight = true;
    }
    if (ch->GetBranch("is_signal")) {
      ch->SetBranchAddress("is_signal", &is_signal);
      has_trident_truth = true;
      if (ch->GetBranch("inv_mass_2mu")) ch->SetBranchAddress("inv_mass_2mu", &inv_mass_2mu);
      if (ch->GetBranch("opening_angle_mrad")) ch->SetBranchAddress("opening_angle_mrad", &opening_angle_mrad);
      if (ch->GetBranch("p_mu_in")) ch->SetBranchAddress("p_mu_in", &p_mu_in);
      if (ch->GetBranch("p_mu_minus")) ch->SetBranchAddress("p_mu_minus", &p_mu_minus);
      if (ch->GetBranch("p_mu_plus")) ch->SetBranchAddress("p_mu_plus", &p_mu_plus);
      if (ch->GetBranch("vtx_x")) ch->SetBranchAddress("vtx_x", &vtx_x);
      if (ch->GetBranch("vtx_y")) ch->SetBranchAddress("vtx_y", &vtx_y);
      if (ch->GetBranch("vtx_z")) ch->SetBranchAddress("vtx_z", &vtx_z);
      if (ch->GetBranch("energy_asym")) ch->SetBranchAddress("energy_asym", &energy_asym);
      if (ch->GetBranch("pt_2mu")) ch->SetBranchAddress("pt_2mu", &pt_2mu);
    }
  }

  // Set up cuts
  std::cout << "Starting cut set up" << std::endl;

  std::vector< snd::trident_cuts::baseCut * > cutFlow;

  int selected_cutset = std::atoi(argv[3]);
  bool is_trident_cutset = (selected_cutset == tridentSelection || 
                            selected_cutset == looseTridentCuts || 
                            selected_cutset == rockTridentPreselection);

  if (selected_cutset == stage1cuts){ // Stage 1 cuts
    cutFlow.push_back( new snd::trident_cuts::avgSciFiFiducialCut(200, 1200, 300, 128*12-200, ch)); // E. Average SciFi hit channel number must be within [200, 1200] (ver) and [300, max-200] (hor)
    cutFlow.push_back( new snd::trident_cuts::avgDSFiducialCut(70, 105, 10, 50, ch)); // F. Average DS hit bar number must be within [70, 105] (ver) and [10, 50] (hor)
    cutFlow.push_back( new snd::trident_cuts::vetoCut(ch)); // B. No veto hits
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0., std::vector<int>(1, 1), ch)); // C. No hits in first SciFi plane
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0., std::vector<int>(1, 2), ch)); // C. No hits in second SciFi plane
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0.05, std::vector<int>(1, 5), ch)); // D. Vertex not in 5th wall
    cutFlow.push_back( new snd::trident_cuts::minSciFiConsecutivePlanes(ch)); // G. At least two consecutive SciFi planes hit
    cutFlow.push_back( new snd::trident_cuts::DSActivityCut(ch)); // H. If there is a downstream hit, require hits in all upstream stations.
    if (not isMC) cutFlow.push_back( new snd::trident_cuts::eventDeltatCut(-1, 100, ch)); // J. Previous event more than 100 clock cycles away. To avoid deadtime issues.

  } else if (selected_cutset == stage1cutsVetoFirst){ // Stage 1 cuts but with Veto cut upfront. For neutral hadron background estimation
    cutFlow.push_back( new snd::trident_cuts::vetoCut(ch)); // B. No veto hits
    cutFlow.push_back( new snd::trident_cuts::avgSciFiFiducialCut(200, 1200, 300, 128*12-200, ch)); // E. Average SciFi hit channel number must be within [200, 1200] (ver) and [300, max-200] (hor)
    cutFlow.push_back( new snd::trident_cuts::avgDSFiducialCut(70, 105, 10, 50, ch)); // F. Average DS hit bar number must be within [70, 105] (ver) and [10, 50] (hor)
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0., std::vector<int>(1, 1), ch)); // C. No hits in first SciFi plane
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0., std::vector<int>(1, 2), ch)); // C. No hits in second SciFi plane
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0.05, std::vector<int>(1, 5), ch)); // D. Vertex not in 5th wall
    cutFlow.push_back( new snd::trident_cuts::minSciFiConsecutivePlanes(ch)); // G. At least two consecutive SciFi planes hit
    cutFlow.push_back( new snd::trident_cuts::DSActivityCut(ch)); // H. If there is a downstream hit, require hits in all upstream stations.
    if (not isMC) cutFlow.push_back( new snd::trident_cuts::eventDeltatCut(-1, 100, ch)); // J. Previous event more than 100 clock cycles away. To avoid deadtime issues.

  } else if (selected_cutset == novetocuts) {
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0.05, std::vector<int>(1, 5), ch)); // D. Vertex not in 5th wall
    cutFlow.push_back( new snd::trident_cuts::avgSciFiFiducialCut(200, 1200, 300, 128*12-200, ch)); // E. Average SciFi hit channel number must be within [200, 1200] (ver) and [300, max-200] (hor)
    cutFlow.push_back( new snd::trident_cuts::avgDSFiducialCut(70, 105, 10, 50, ch)); // F. Average DS hit bar number must be within [70, 105] (ver) and [10, 50] (hor)
    cutFlow.push_back( new snd::trident_cuts::minSciFiConsecutivePlanes(ch)); // G. At least two consecutive SciFi planes hit
    cutFlow.push_back( new snd::trident_cuts::DSActivityCut(ch)); // H. If there is a downstream hit, require hits in all upstream stations.
    if (not isMC) cutFlow.push_back( new snd::trident_cuts::eventDeltatCut(-1, 100, ch)); // J. Previous event more than 100 clock cycles away. To avoid deadtime issues.

  } else if (selected_cutset == FVsideband){
    cutFlow.push_back( new snd::trident_cuts::vetoCut(ch)); // B. No veto hits
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0., std::vector<int>(1, 1), ch)); // C. No hits in first SciFi plane
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0., std::vector<int>(1, 2), ch)); // D. Vertex not in 5th wall
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0.05, std::vector<int>(1, 5), ch)); // D. Vertex not in 5th wall
    cutFlow.push_back( new snd::trident_cuts::avgSciFiFiducialCut(200, 1200, 300, 128*12-200, ch, true)); // E. Average SciFi hit channel number must be within [200, 1200] (ver) and [300, max-200] (hor)
    cutFlow.push_back( new snd::trident_cuts::minSciFiConsecutivePlanes(ch)); // G. At least two consecutive SciFi planes hit
    cutFlow.push_back( new snd::trident_cuts::DSActivityCut(ch)); // H. If there is a downstream hit, require hits in all upstream stations.
    if (not isMC) cutFlow.push_back( new snd::trident_cuts::eventDeltatCut(-1, 100, ch)); // J. Previous event more than 100 clock cycles away. To avoid deadtime issues.

  } else if (selected_cutset == allowWalls2and5) {
    cutFlow.push_back( new snd::trident_cuts::avgSciFiFiducialCut(200, 1200, 300, 128*12-200, ch)); // E. Average SciFi hit channel number must be within [200, 1200] (ver) and [300, max-200] (hor)
    cutFlow.push_back( new snd::trident_cuts::avgDSFiducialCut(70, 105, 10, 50, ch)); // F. Average DS hit bar number must be within [70, 105] (ver) and [10, 50] (hor)
    cutFlow.push_back( new snd::trident_cuts::vetoCut(ch)); // B. No veto hits
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0., std::vector<int>(1, 1), ch)); // C. No hits in first SciFi plane
    cutFlow.push_back( new snd::trident_cuts::DSActivityCut(ch)); // H. If there is a downstream hit, require hits in all upstream stations.
    if (not isMC) cutFlow.push_back( new snd::trident_cuts::eventDeltatCut(-1, 100, ch)); // J. Previous event more than 100 clock cycles away. To avoid deadtime issues.

  } else if (selected_cutset == nueFilter) {
    cutFlow.push_back( new snd::trident_cuts::avgSciFiFiducialCut(200, 1200, 300, 128*12-200, ch)); // E. Average SciFi hit channel number must be within [200, 1200] (ver) and [300, max-200] (hor)
    cutFlow.push_back( new snd::trident_cuts::vetoCut(ch)); // B. No veto hits
    cutFlow.push_back( new snd::trident_cuts::sciFiStationCut(0.05, std::vector<int>(1, 5), ch)); // D. Vertex not in 5th wall
    cutFlow.push_back( new snd::trident_cuts::DSVetoCut(ch)); // D. Veto events with hits in last DS planes
    if (not isMC) cutFlow.push_back( new snd::trident_cuts::eventDeltatCut(-1, 100, ch)); // J. Previous event more than 100 clock cycles away. To avoid deadtime issues.
  } else if (selected_cutset == tridentSelection) {
    cutFlow.push_back( new snd::trident_cuts::hasVetoHitsCut(ch)); // Require veto hits
    cutFlow.push_back( new snd::trident_cuts::minSciFiPlanesCut(3, 3, ch)); // Require at least 3 planes in each projection
    cutFlow.push_back( new snd::trident_cuts::tridentHitsCut(9, 3, ch)); // Require 9/3 or 3/9 hits
    cutFlow.push_back( new snd::trident_cuts::maxSciFiHitsCut(200, ch)); // Total hits < 200
    cutFlow.push_back( new snd::trident_cuts::maxPlaneSciFiHitsCut(100, ch)); // Max hits in any plane < 100
    cutFlow.push_back( new snd::trident_cuts::maxSciFiSignalCut(400, ch)); // Total signal < 400
    cutFlow.push_back( new snd::trident_cuts::maxPlaneSciFiSignalCut(250, ch)); // Max signal in any plane < 100
    cutFlow.push_back( new snd::trident_cuts::tridentDensityCut(40, 5000, ch)); // Sum density < 5000
    cutFlow.push_back( new snd::trident_cuts::avgSciFiFiducialCut(200, 1200, 300, 128*12-200, ch)); 
    cutFlow.push_back( new snd::trident_cuts::avgDSFiducialCut(70, 105, 10, 50, ch)); 
    if (not isMC) cutFlow.push_back( new snd::trident_cuts::eventDeltatCut(-1, 100, ch)); 
  } else if (selected_cutset == looseTridentCuts) {
    cutFlow.push_back( new snd::trident_cuts::DSQDCCut(400.0, ch)); // 1. Total DS QDC >= 400
    cutFlow.push_back( new snd::trident_cuts::minDSHitsCut(8, ch)); // 2. Minimum 8 DS hits
    cutFlow.push_back( new snd::trident_cuts::lastDSPlaneCut(2, ch)); // 3. Last active DS plane is 3rd or 4th (plane index >= 2)
    cutFlow.push_back( new snd::trident_cuts::minSciFiHits(10, ch)); // 4. Total SciFi hits >= 10
    if (not isMC) cutFlow.push_back( new snd::trident_cuts::eventDeltatCut(-1, 100, ch)); // 5. Previous event > 100 clock cycles away
  } else if (selected_cutset == rockTridentPreselection) {
    cutFlow.push_back( new snd::trident_cuts::minDSHitsCut(8, ch));             // 1) ds_nhits >= 8
    cutFlow.push_back( new snd::trident_cuts::minUSHitsCut(5, ch));             // 2) us_nhits >= 5
    cutFlow.push_back( new snd::trident_cuts::tridentHitsCut(9, 3, ch));        // 3) 9/3 or 3/9 scifi hits
    cutFlow.push_back( new snd::trident_cuts::minMaxSciFiSignalCut(5.0, ch));   // 4) scifi_max_qdc >= 5
    cutFlow.push_back( new snd::trident_cuts::DSQDCCut(500.0, ch));             // 6) ds_sum_qdc >= 500
    cutFlow.push_back( new snd::trident_cuts::USQDCCut(40.0, ch));              // 7) us_sum_qdc >= 40
    cutFlow.push_back( new snd::trident_cuts::minMaxUSQDCCut(7.5, ch));         // 8) us_max_qdc >= 7.5
    cutFlow.push_back( new snd::trident_cuts::DStoSciFiQDCRatioCut(0.05, ch));  // 9) ratio_ds_to_scifi_qdc >= 0.05
    cutFlow.push_back( new snd::trident_cuts::lastDSPlaneCut(3, ch));           // 10) ds_deepest_station == 4 (station index >= 3)
    if (not isMC) cutFlow.push_back( new snd::trident_cuts::eventDeltatCut(-1, 100, ch)); // 11) event 100 clock cycles away (data only)
  } else {
    std::cout << "Unrecognized cutset. Exitting" << std::endl;
    return -1;
  }
  std::cout << "Done initializing cuts" << std::endl;

  int n_cuts = (int) cutFlow.size();

  // Output file
  TFile * outFile = new TFile(argv[2], "RECREATE");
  std::cout << "Got output file" << std::endl;

  ch->GetEntry(0);
  if (ch->GetFile()->Get("BranchList")) ch->GetFile()->Get("BranchList")->Write("BranchList", TObject::kSingleKey);
  if (ch->GetFile()->Get("TimeBasedBranchList")) ch->GetFile()->Get("TimeBasedBranchList")->Write("TimeBasedBranchList", TObject::kSingleKey);
  if (ch->GetFile()->Get("FileHeader")) ch->GetFile()->Get("FileHeader")->Write();
  if (ch->GetFile()->Get("FileHeaderHeader")) ch->GetFile()->Get("FileHeaderHeader")->Write();

  // Set up all branches to copy to output TTree.
  TTree * outTree = ch->CloneTree(0);
  std::cout << "Got output tree" << std::endl;

  // Book histograms
  // Cut-by-cut
  // All cut variables (unweighted and weighted)
  std::vector<std::vector<TH1D*> > cut_by_cut_var_histos = std::vector<std::vector<TH1D*> >();
  std::vector<std::vector<TH1D*> > cut_by_cut_var_weighted_histos = std::vector<std::vector<TH1D*> >();

  for (int i_cut = -1; i_cut < n_cuts; i_cut++){
    std::vector<TH1D*> this_cut_by_cut_var_histos = std::vector<TH1D*>();
    std::vector<TH1D*> this_cut_by_cut_var_w_histos = std::vector<TH1D*>();
    for (snd::trident_cuts::baseCut * cut : cutFlow) {
      for(int i_dim = 0; i_dim < cut->getNbins().size(); i_dim++){
	this_cut_by_cut_var_histos.push_back(new TH1D((std::to_string(i_cut)+"_"+cut->getShortName()+"_"+std::to_string(i_dim)).c_str(),
						      cut->getShortName().c_str(),
						      cut->getNbins()[i_dim], cut->getRangeStart()[i_dim], cut->getRangeEnd()[i_dim]));
	if (has_mc_weight) {
	  this_cut_by_cut_var_w_histos.push_back(new TH1D((std::to_string(i_cut)+"_"+cut->getShortName()+"_"+std::to_string(i_dim)+"_weighted").c_str(),
							 (cut->getShortName() + " (weighted)").c_str(),
							 cut->getNbins()[i_dim], cut->getRangeStart()[i_dim], cut->getRangeEnd()[i_dim]));
	}
      }
    }
    cut_by_cut_var_histos.push_back(this_cut_by_cut_var_histos);
    if (has_mc_weight) cut_by_cut_var_weighted_histos.push_back(this_cut_by_cut_var_w_histos);
  }

  // Neutrino MC truth histograms (only booked for neutrino cutsets 0..5)
  std::vector<std::vector<std::vector<TH1D*> > > cut_by_cut_truth_histos = std::vector<std::vector<std::vector<TH1D*> > >();
  if (isMC && !is_trident_cutset) {
    for (int i_species = 0; i_species < 5; i_species++){ // e, mu, tau, NC, Other
      std::vector<std::vector<TH1D*> > this_species_histos = std::vector<std::vector<TH1D*> >();
      std::string species_suffix;
      switch (i_species) {
      case 0:
	species_suffix = "nueCC";
	break;
      case 1:
	species_suffix = "numuCC";
	break;
      case 2:
	species_suffix = "nutauCC";
	break;
      case 3:
	species_suffix = "NC";
	break;
      case 4:
	species_suffix = "Other"; // For PG sim
	break;
      default :
	std::cerr << "MC truth histograms initialization error! Unknown species" << std::endl;
	exit(-1);
      }

      for (int i_cut = -1; i_cut < n_cuts; i_cut++){
	std::vector<TH1D*> this_cut_by_cut_truth_histos = std::vector<TH1D*>();
	this_cut_by_cut_truth_histos.push_back(new TH1D((species_suffix+"_"+std::to_string(i_cut)+"_Enu").c_str(), "Enu", 300, 0, 3000));
	this_cut_by_cut_truth_histos.push_back(new TH1D((species_suffix+"_"+std::to_string(i_cut)+"_EEM").c_str(), "ELep", 300, 0, 3000));
	this_cut_by_cut_truth_histos.push_back(new TH1D((species_suffix+"_"+std::to_string(i_cut)+"_EHad").c_str(), "EHad", 300, 0, 3000));
	this_cut_by_cut_truth_histos.push_back(new TH1D((species_suffix+"_"+std::to_string(i_cut)+"_vtxX").c_str(), "vtxX", 200, -100, 0));
	this_cut_by_cut_truth_histos.push_back(new TH1D((species_suffix+"_"+std::to_string(i_cut)+"_vtxY").c_str(), "vtxY", 200, 0, 100));
	this_cut_by_cut_truth_histos.push_back(new TH1D((species_suffix+"_"+std::to_string(i_cut)+"_vtxZ").c_str(), "vtxZ", 200, 280, 380));

	this_species_histos.push_back(this_cut_by_cut_truth_histos);
      }
      cut_by_cut_truth_histos.push_back(this_species_histos);
    }
  }

  // Trident MCTruth histograms: [cut_index + 1][var_index] (only booked for trident cutsets 6..8)
  std::vector<std::vector<TH1D*>> cut_by_cut_trident_truth_histos;
  std::vector<std::vector<TH1D*>> cut_by_cut_trident_truth_weighted_histos;

  if (isMC && is_trident_cutset && has_trident_truth) {
    for (int i_cut = -1; i_cut < n_cuts; i_cut++) {
      std::string c_str = std::to_string(i_cut);
      std::vector<TH1D*> this_cut_trident_histos;
      std::vector<TH1D*> this_cut_trident_w_histos;

      // 0: inv_mass_2mu
      this_cut_trident_histos.push_back(new TH1D(("trident_" + c_str + "_inv_mass_2mu").c_str(), "M_{#mu#mu};M_{#mu#mu} [GeV];Events", 100, 0.0, 2.5));
      // 1: opening_angle_mrad
      this_cut_trident_histos.push_back(new TH1D(("trident_" + c_str + "_opening_angle").c_str(), "#theta_{#mu#mu};#theta_{#mu#mu} [mrad];Events", 100, 0.0, 50.0));
      // 2: p_mu_in
      this_cut_trident_histos.push_back(new TH1D(("trident_" + c_str + "_p_mu_in").c_str(), "p_{#mu}^{in};p_{#mu}^{in} [GeV];Events", 150, 0.0, 3000.0));
      // 3: p_mu_minus
      this_cut_trident_histos.push_back(new TH1D(("trident_" + c_str + "_p_mu_minus").c_str(), "p_{#mu^{-}};p_{#mu^{-}} [GeV];Events", 150, 0.0, 1500.0));
      // 4: p_mu_plus
      this_cut_trident_histos.push_back(new TH1D(("trident_" + c_str + "_p_mu_plus").c_str(), "p_{#mu^{+}};p_{#mu^{+}} [GeV];Events", 150, 0.0, 1500.0));
      // 5: vtx_z
      this_cut_trident_histos.push_back(new TH1D(("trident_" + c_str + "_vtxZ").c_str(), "Z_{vtx};Z_{vtx} [cm];Events", 250, -4000.0, 1000.0));
      // 6: vtx_x
      this_cut_trident_histos.push_back(new TH1D(("trident_" + c_str + "_vtxX").c_str(), "X_{vtx};X_{vtx} [cm];Events", 100, -100.0, 100.0));
      // 7: vtx_y
      this_cut_trident_histos.push_back(new TH1D(("trident_" + c_str + "_vtxY").c_str(), "Y_{vtx};Y_{vtx} [cm];Events", 100, -100.0, 100.0));
      // 8: energy_asym
      this_cut_trident_histos.push_back(new TH1D(("trident_" + c_str + "_energy_asym").c_str(), "A_{E};(E_{#mu^{-}}-E_{#mu^{+}})/(E_{#mu^{-}}+E_{#mu^{+}});Events", 50, -1.0, 1.0));
      // 9: pt_2mu
      this_cut_trident_histos.push_back(new TH1D(("trident_" + c_str + "_pt_2mu").c_str(), "p_{T}^{#mu#mu};p_{T}^{#mu#mu} [GeV];Events", 100, 0.0, 5.0));

      if (has_mc_weight) {
        this_cut_trident_w_histos.push_back(new TH1D(("trident_" + c_str + "_inv_mass_2mu_weighted").c_str(), "M_{#mu#mu} (weighted);M_{#mu#mu} [GeV];Weighted Events", 100, 0.0, 2.5));
        this_cut_trident_w_histos.push_back(new TH1D(("trident_" + c_str + "_opening_angle_weighted").c_str(), "#theta_{#mu#mu} (weighted);#theta_{#mu#mu} [mrad];Weighted Events", 100, 0.0, 50.0));
        this_cut_trident_w_histos.push_back(new TH1D(("trident_" + c_str + "_p_mu_in_weighted").c_str(), "p_{#mu}^{in} (weighted);p_{#mu}^{in} [GeV];Weighted Events", 150, 0.0, 3000.0));
        this_cut_trident_w_histos.push_back(new TH1D(("trident_" + c_str + "_p_mu_minus_weighted").c_str(), "p_{#mu^{-}} (weighted);p_{#mu^{-}} [GeV];Weighted Events", 150, 0.0, 1500.0));
        this_cut_trident_w_histos.push_back(new TH1D(("trident_" + c_str + "_p_mu_plus_weighted").c_str(), "p_{#mu^{+}} (weighted);p_{#mu^{+}} [GeV];Weighted Events", 150, 0.0, 1500.0));
        this_cut_trident_w_histos.push_back(new TH1D(("trident_" + c_str + "_vtxZ_weighted").c_str(), "Z_{vtx} (weighted);Z_{vtx} [cm];Weighted Events", 250, -4000.0, 1000.0));
        this_cut_trident_w_histos.push_back(new TH1D(("trident_" + c_str + "_vtxX_weighted").c_str(), "X_{vtx} (weighted);X_{vtx} [cm];Weighted Events", 100, -100.0, 100.0));
        this_cut_trident_w_histos.push_back(new TH1D(("trident_" + c_str + "_vtxY_weighted").c_str(), "Y_{vtx} (weighted);Y_{vtx} [cm];Weighted Events", 100, -100.0, 100.0));
        this_cut_trident_w_histos.push_back(new TH1D(("trident_" + c_str + "_energy_asym_weighted").c_str(), "A_{E} (weighted);(E_{#mu^{-}}-E_{#mu^{+}})/(E_{#mu^{-}}+E_{#mu^{+}});Weighted Events", 50, -1.0, 1.0));
        this_cut_trident_w_histos.push_back(new TH1D(("trident_" + c_str + "_pt_2mu_weighted").c_str(), "p_{T}^{#mu#mu} (weighted);p_{T}^{#mu#mu} [GeV];Weighted Events", 100, 0.0, 5.0));
      }

      cut_by_cut_trident_truth_histos.push_back(this_cut_trident_histos);
      if (has_mc_weight) cut_by_cut_trident_truth_weighted_histos.push_back(this_cut_trident_w_histos);
    }
  }
  // N-1
  std::vector<TH1D*> n_minus_1_var_histos = std::vector<TH1D*>();
  std::vector<TH1D*> n_minus_1_var_weighted_histos = std::vector<TH1D*>();
  for (snd::trident_cuts::baseCut * cut : cutFlow) {
    for(int i_dim = 0; i_dim < cut->getNbins().size(); i_dim++){
      n_minus_1_var_histos.push_back(new TH1D(("n_minus_1_"+cut->getShortName()+"_"+std::to_string(i_dim)).c_str(),
					      cut->getShortName().c_str(),
					      cut->getNbins()[i_dim], cut->getRangeStart()[i_dim], cut->getRangeEnd()[i_dim]));
      if (has_mc_weight) {
	n_minus_1_var_weighted_histos.push_back(new TH1D(("n_minus_1_"+cut->getShortName()+"_"+std::to_string(i_dim)+"_weighted").c_str(),
							(cut->getShortName() + " (weighted)").c_str(),
							cut->getNbins()[i_dim], cut->getRangeStart()[i_dim], cut->getRangeEnd()[i_dim]));
      }
    }
  }

  std::cout << "Done initializing histograms" << std::endl;

  // Cut flow
  TH1D * cutFlowHistogram = new TH1D("cutFlow", "Cut flow;;Number of events passing cut", n_cuts+1, 0, n_cuts+1);
  for (int i = 2; i <= cutFlowHistogram->GetNbinsX(); i++){
    cutFlowHistogram->GetXaxis()->SetBinLabel(i, cutFlow.at(i-2)->getName().c_str());
  }

  TH1D * cutFlowWeightedHistogram = nullptr;
  if (has_mc_weight) {
    cutFlowWeightedHistogram = new TH1D("cutFlow_weighted", "Cut flow (weighted);;Weighted events passing cut", n_cuts+1, 0, n_cuts+1);
    for (int i = 2; i <= cutFlowWeightedHistogram->GetNbinsX(); i++){
      cutFlowWeightedHistogram->GetXaxis()->SetBinLabel(i, cutFlow.at(i-2)->getName().c_str());
    }
  }

  std::cout << "Done initializing cut flow histogram" << std::endl;

  // Get number of entries
  unsigned long int n_entries = ch->GetEntries();

  // Holder for cut results
  std::vector<bool> passes_cut = std::vector<bool>(n_cuts, false);

  std::cout << "Starting event loop" << std::endl;
  for (unsigned long int i_entry = 0; i_entry < n_entries; i_entry++){
    ch->GetEntry(i_entry);
    if (i_entry % 10000 == 0) std::cout << "Reading entry " << i_entry << " / " << n_entries << std::endl;

    cutFlowHistogram->Fill(0);
    if (cutFlowWeightedHistogram) cutFlowWeightedHistogram->Fill((Double_t) 0, mc_weight);

    // Apply cuts
    int n_cuts_passed = 0;
    bool accept_event = true;
    int i_cut = 0;
    for (snd::trident_cuts::baseCut * cut : cutFlow){
      if (cut->passCut()){
	if (accept_event) {
	  cutFlowHistogram->Fill(1 + i_cut);
	  if (cutFlowWeightedHistogram) cutFlowWeightedHistogram->Fill((Double_t) (1 + i_cut), mc_weight);
	}
	passes_cut[i_cut] = true;
	n_cuts_passed++;
      } else {
	accept_event = false;
	passes_cut[i_cut] = false;
      }
      i_cut++;
    }
    if (accept_event) outTree->Fill();


    // Fill histograms
    std::vector<TH1D*>::iterator hist_it;
    std::vector<TH1D*>::iterator hist_w_it;
    // Sequential
    for (int seq_cut = -1; seq_cut < ((int) passes_cut.size()); seq_cut++){
      if (seq_cut >= 0){
	if (not passes_cut[seq_cut]) break;
      }
      hist_it = cut_by_cut_var_histos[seq_cut+1].begin();
      if (has_mc_weight) hist_w_it = cut_by_cut_var_weighted_histos[seq_cut+1].begin();

      for (snd::trident_cuts::baseCut * cut : cutFlow) {
	for (int i_dim = 0; i_dim < cut->getPlotVar().size(); i_dim++){
	  (*hist_it)->Fill(cut->getPlotVar()[i_dim]);
	  hist_it++;
	  if (has_mc_weight) {
	    (*hist_w_it)->Fill(cut->getPlotVar()[i_dim], mc_weight);
	    hist_w_it++;
	  }
	}
      }

      if (isMC && !is_trident_cutset) {
	int this_species = 4; // Default to 'Other' (4)
	int n_tracks = MCTracks ? MCTracks->GetEntries() : 0;
	if (n_tracks >= 2) {
	  ShipMCTrack * track0 = (ShipMCTrack*) MCTracks->At(0);
	  ShipMCTrack * track1 = (ShipMCTrack*) MCTracks->At(1);
	  if (track0 && track1) {
	    int pdgIn = abs(track0->GetPdgCode());
	    int pdgOut = abs(track1->GetPdgCode());

	    if ((pdgIn == 12 && pdgOut == 11) || (pdgIn == 14 && pdgOut == 13) || (pdgIn == 16 && pdgOut == 15)) {
	      // CC
	      if (pdgIn == 12) this_species = 0; // nueCC
	      else if (pdgIn == 14) this_species = 1; // numuCC
	      else if (pdgIn == 16) this_species = 2; // nutauCC
	    } else if (pdgIn == pdgOut && (pdgIn == 12 || pdgIn == 14 || pdgIn == 16)) {
	      // NC
	      this_species = 3;
	    } else {
	      // Other
	      this_species = 4;
	    }
	  }
	}

	if (n_tracks > 0) {
	  ShipMCTrack * track0 = (ShipMCTrack*) MCTracks->At(0);
	  if (track0) {
	    cut_by_cut_truth_histos[this_species][seq_cut+1][0]->Fill(track0->GetEnergy()); // Enu / incoming energy
	    if (this_species < 4 && n_tracks >= 2) {
	      ShipMCTrack * track1 = (ShipMCTrack*) MCTracks->At(1);
	      if (track1) {
		cut_by_cut_truth_histos[this_species][seq_cut+1][1]->Fill(track1->GetEnergy()); // ELep
		cut_by_cut_truth_histos[this_species][seq_cut+1][2]->Fill(track0->GetEnergy() - track1->GetEnergy()); // EHad
	      }
	    }
	    cut_by_cut_truth_histos[this_species][seq_cut+1][3]->Fill(track0->GetStartX()); // X
	    cut_by_cut_truth_histos[this_species][seq_cut+1][4]->Fill(track0->GetStartY()); // Y
	    cut_by_cut_truth_histos[this_species][seq_cut+1][5]->Fill(track0->GetStartZ()); // Z
	  }
	}
      }

      if (isMC && is_trident_cutset && has_trident_truth && is_signal) {
	cut_by_cut_trident_truth_histos[seq_cut+1][0]->Fill(inv_mass_2mu);
	cut_by_cut_trident_truth_histos[seq_cut+1][1]->Fill(opening_angle_mrad);
	cut_by_cut_trident_truth_histos[seq_cut+1][2]->Fill(p_mu_in);
	cut_by_cut_trident_truth_histos[seq_cut+1][3]->Fill(p_mu_minus);
	cut_by_cut_trident_truth_histos[seq_cut+1][4]->Fill(p_mu_plus);
	cut_by_cut_trident_truth_histos[seq_cut+1][5]->Fill(vtx_z);
	cut_by_cut_trident_truth_histos[seq_cut+1][6]->Fill(vtx_x);
	cut_by_cut_trident_truth_histos[seq_cut+1][7]->Fill(vtx_y);
	cut_by_cut_trident_truth_histos[seq_cut+1][8]->Fill(energy_asym);
	cut_by_cut_trident_truth_histos[seq_cut+1][9]->Fill(pt_2mu);

	if (has_mc_weight) {
	  cut_by_cut_trident_truth_weighted_histos[seq_cut+1][0]->Fill(inv_mass_2mu, mc_weight);
	  cut_by_cut_trident_truth_weighted_histos[seq_cut+1][1]->Fill(opening_angle_mrad, mc_weight);
	  cut_by_cut_trident_truth_weighted_histos[seq_cut+1][2]->Fill(p_mu_in, mc_weight);
	  cut_by_cut_trident_truth_weighted_histos[seq_cut+1][3]->Fill(p_mu_minus, mc_weight);
	  cut_by_cut_trident_truth_weighted_histos[seq_cut+1][4]->Fill(p_mu_plus, mc_weight);
	  cut_by_cut_trident_truth_weighted_histos[seq_cut+1][5]->Fill(vtx_z, mc_weight);
	  cut_by_cut_trident_truth_weighted_histos[seq_cut+1][6]->Fill(vtx_x, mc_weight);
	  cut_by_cut_trident_truth_weighted_histos[seq_cut+1][7]->Fill(vtx_y, mc_weight);
	  cut_by_cut_trident_truth_weighted_histos[seq_cut+1][8]->Fill(energy_asym, mc_weight);
	  cut_by_cut_trident_truth_weighted_histos[seq_cut+1][9]->Fill(pt_2mu, mc_weight);
	}
      }
    }

    // N-1
    int current_cut = 0;
    hist_it = n_minus_1_var_histos.begin();
    if (has_mc_weight) hist_w_it = n_minus_1_var_weighted_histos.begin();

    for (snd::trident_cuts::baseCut * cut : cutFlow) {
      for (int i_dim = 0; i_dim < cut->getPlotVar().size(); i_dim++){
	if (((not passes_cut[current_cut]) and (n_cuts_passed == (cutFlow.size()-1)))
	    or (n_cuts_passed == cutFlow.size()) ) {
	  (*hist_it)->Fill(cut->getPlotVar()[i_dim]);
	  if (has_mc_weight) (*hist_w_it)->Fill(cut->getPlotVar()[i_dim], mc_weight);
	}
	hist_it++;
	if (has_mc_weight) hist_w_it++;
      }
      current_cut++;
    }
  }

  outFile->Write();
  outFile->Close();

  _Exit(0);
}
