#pragma once

#include <vector>

#include "sndBaseCut.h"

#include "SNDLHCEventHeader.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
  
    class EventHeaderBaseCut : public snd::trident_cuts::baseCut {

    protected :
      static SNDLHCEventHeader * header;
      static TChain * tree;

      EventHeaderBaseCut(TChain * ch);
      ~EventHeaderBaseCut(){;}
    };

  }
}
