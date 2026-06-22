#ifndef EVENT_TYPE_H
#define EVENT_TYPE_H

#include "TNamed.h"
#include <map>
#include <string>

class Event_Type : public TNamed {
public:
  Event_Type() : TNamed() {}
  Event_Type(const char* name, const char* title) : TNamed(name, title) {}
  virtual ~Event_Type() {}

  std::map<std::string, std::string> data;

  ClassDef(Event_Type, 1)
};

#endif // EVENT_TYPE_H
