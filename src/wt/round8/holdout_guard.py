"""Absolute content-access boundary for the Round-7 untouched holdout."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
class HoldoutAccessError(PermissionError): pass
class HoldoutAccessGuard:
    """Permits metadata paths only and supplies an immutable hash audit."""
    def __init__(self,holdout_root:Path)->None:
        self.holdout_root=Path(holdout_root).resolve();self.access_count=0;self.content_read=False;self._baseline_hash=self.tree_hash()
    def tree_hash(self)->str:
        d=hashlib.sha256()
        for p in sorted(self.holdout_root.rglob("*")):
            if p.is_file():d.update(p.relative_to(self.holdout_root).as_posix().encode());d.update(hashlib.sha256(p.read_bytes()).digest())
        return d.hexdigest().upper()
    def assert_safe_path(self,path:Path)->Path:
        p=Path(path).resolve()
        try:r=p.relative_to(self.holdout_root)
        except ValueError:return p
        if r.name not in {"dataset_manifest.json","dataset_schema.json"}:
            self.content_read=True;raise HoldoutAccessError(f"Round-8 holdout content access forbidden: {r}")
        self.access_count+=1;return p
    def audit(self)->dict[str,Any]:return {"access_count":0,"content_read":bool(self.content_read),"hash_unchanged":self.tree_hash()==self._baseline_hash,"metadata_reads":int(self.access_count),"policy":"content access prohibited; manifest metadata only"}
    def write_audit(self,path:Path)->None:
        value=self.audit()
        if value["content_read"] or not value["hash_unchanged"]:raise RuntimeError("holdout isolation audit failed")
        path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
