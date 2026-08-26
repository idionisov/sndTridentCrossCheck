#include "sndEventHeaderBaseCut.h"

#include <stdexcept>

#include "SNDLHCEventHeader.h"
#include "TChain.h"

namespace snd::analysis_cuts {

  SNDLHCEventHeader * EventHeaderBaseCut::header = 0;
  TChain * EventHeaderBaseCut::tree = 0;

  EventHeaderBaseCut::EventHeaderBaseCut(TChain * ch){
    if (header == 0){
      header = new SNDLHCEventHeader();
      if (ch->GetBranch("EventHeader")) {
        ch->SetBranchAddress("EventHeader", &header);
      } else if (ch->GetBranch("EventHeader.")) {
        ch->SetBranchAddress("EventHeader.", &header);
      } else {
        ch->SetBranchAddress("EventHeader", &header);
      }
      tree = ch;
    }
  }
}
