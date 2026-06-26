#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "TSystem.h"

// SNDLHC headers
#include "sndScifiHit.h"
#include "SNDLHCEventHeader.h"

void perPlane(std::string outputFileName, std::vector<std::string> inputFiles) {
    // Load required libraries
    gSystem->Load("libBase");
    gSystem->Load("libShipData");
    gSystem->Load("libshipLHC");
    gSystem->Load("libsnd_analysis_tools");

    TChain* chain = new TChain("cbmsim");

    // 1. Let TChain handle the wildcard expansion first
    for (const auto& file : inputFiles) {
        chain->Add(file.c_str());
    }

    // 2. Extract the first successfully resolved file to check the tree name
    TObjArray* fileElements = chain->GetListOfFiles();
    if (fileElements->GetEntries() > 0) {
        // TChainElement contains the actual resolved file path
        TChainElement* element = (TChainElement*)fileElements->At(0);
        TFile* firstFile = TFile::Open(element->GetTitle());
        if (firstFile) {
            if (!firstFile->Get("cbmsim")) {
                chain->SetName("rawConv");
            }
            firstFile->Close();
        }
    } else {
        std::cerr << "Error: No files found matching the input." << std::endl;
        return;
    }

    long nEntries = chain->GetEntries();
    std::cout << "Processing " << nEntries << " entries from tree: " << chain->GetName() << std::endl;

    TClonesArray* scifiHits = nullptr;
    SNDLHCEventHeader* eventHeader = nullptr;
    chain->SetBranchAddress("Digi_ScifiHits", &scifiHits);

    bool isData = (std::string(chain->GetName()) == "rawConv");
    if (isData) {
        chain->SetBranchAddress("EventHeader", &eventHeader);
    }

    // Set up output file and tree
    TFile* outFile = new TFile(outputFileName.c_str(), "RECREATE");
    TTree* outTree = new TTree("perPlane", "SciFi Per Plane Data");

    int out_event;
    int out_plane_id;
    int out_nhits;
    double out_qdc;

    outTree->Branch("event_number", &out_event);
    outTree->Branch("plane_id", &out_plane_id);
    outTree->Branch("nhits", &out_nhits);
    outTree->Branch("qdc", &out_qdc);

    for (long i = 0; i < nEntries; ++i) {
        chain->GetEntry(i);
        if (i % 10000 == 0) std::cout << "Event " << i << " / " << nEntries << std::endl;

        out_event = eventHeader->GetEventNumber();

        // plane_id -> {nhits, qdc}
        // plane_id formula: station * 10 + isVertical
        std::map<int, std::pair<int, double>> planes;

        int nHits = scifiHits->GetEntries();
        for (int j = 0; j < nHits; ++j) {
            sndScifiHit* hit = (sndScifiHit*)scifiHits->At(j);
            if (!hit->isValid()) continue;

            int plane_id = hit->GetStation() * 10 + (hit->isVertical() ? 1 : 0);
            planes[plane_id].first++;
            planes[plane_id].second += hit->GetSignal();
        }

        for (auto const& [id, data] : planes) {
            out_plane_id = id;
            out_nhits = data.first;
            out_qdc = data.second;
            outTree->Fill();
        }
    }

    outTree->Write();
    outFile->Close();
    std::cout << "Done. Output saved to " << outputFileName << std::endl;
}

// Entry point for running via: root -l -b -q 'perPlane.cxx("out.root", {"in1.root", "in2.root"})'
void perPlane() {
    std::cout << "Usage: root -l 'perPlane.cxx(\"output.root\", {\"input.root\"})'" << std::endl;
}
