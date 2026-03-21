import os
import re
import urllib.request

_PROLOG_DIR = "third_party/geo_eval"
_BASE_URL = "https://raw.githubusercontent.com/berlino/grammar-prompting/main/third_party/geo_eval"
_FILES = ("geobase.pl", "geoquery.pl", "eval.pl")


def _ensure_prolog_files() -> None:
    os.makedirs(_PROLOG_DIR, exist_ok=True)
    for fname in _FILES:
        path = os.path.join(_PROLOG_DIR, fname)
        if not os.path.exists(path):
            url = f"{_BASE_URL}/{fname}"
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, path)


class GeoExecutor:
    def __init__(self):
        _ensure_prolog_files()
        from pyswip import Prolog

        self._prolog = Prolog()
        self._prolog.consult(os.path.join(_PROLOG_DIR, "geobase.pl"))
        self._prolog.consult(os.path.join(_PROLOG_DIR, "geoquery.pl"))
        self._prolog.consult(os.path.join(_PROLOG_DIR, "eval.pl"))

    def execute(self, program: str) -> str | None:
        program = re.sub(r"' (\w+) (\w+) '", "'" + r"\1#\2" + "'", program)
        program = re.sub(r"' (\w+) (\w+) (\w+) '", "'" + r"\1#\2#\3" + "'", program)
        program = program.replace(" ", "").replace("#", " ")

        try:
            answers = list(
                self._prolog.query(f"eval({program}, X).", maxresult=1)
            )
            if not answers:
                return None
            return str([str(a) for a in answers[0]["X"]])
        except Exception:
            return None


_executor: GeoExecutor | None = None


def execute(program: str) -> str | None:
    global _executor
    if _executor is None:
        _executor = GeoExecutor()
    return _executor.execute(program)


def is_available() -> bool:
    try:
        _ensure_prolog_files()
        from pyswip import Prolog

        p = Prolog()
        p.consult(os.path.join(_PROLOG_DIR, "geobase.pl"))
        return True
    except Exception:
        return False
