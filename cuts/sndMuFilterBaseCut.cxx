#include "sndMuFilterBaseCut.h"

#include <vector>

#include "TClonesArray.h"
#include "MuFilterHit.h"

namespace snd::trident_cuts {

  TClonesArray * MuFilterBaseCut::muFilterDigiHitCollection = 0;

  void MuFilterBaseCut::setupBranch(TChain * ch){
    if (muFilterDigiHitCollection == 0){
      muFilterDigiHitCollection = new TClonesArray("MuFilterHit", 470);
      ch->SetBranchAddress("Digi_MuFilterHits", &muFilterDigiHitCollection); 
    }
  }

  MuFilterBaseCut::MuFilterBaseCut(TChain * ch){
    setupBranch(ch);
  }
}
