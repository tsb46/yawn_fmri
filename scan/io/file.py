"""
Module for handling data file paths and iterating subject lists.
"""

import json
from enum import Enum
from typing import Literal, Tuple, Generator, List, Union

import pandas as pd

from scan import utils


class Field(Enum):
    """
    Enum for participant list fields
    """

    SUBJECT = "subject"
    SESSION = "session"


class Participant:
    """
    Class for loading and iterating
    through participant lists.

    Attributes
    ----------
    dataset : str
        chosen dataset
    subj_ses_select : list[tuple[str, str]]
        list of subject and session label pairs to load. If None,
        load all subjects and sessions in the dataset (default: None).
        For subjects with only one session, pass as a tuple with the
        session as None (e.g. ('01', None))

    Methods
    -------
    subject():
        Iterate through subjects (ignore scans)

    subject_by_session():
        Iterate through subjects with individual sessions returned
        as nested list

    filepaths(data, basedir, echo):
        Iterate through filepaths

    to_file(data, subject, session, echo=None):
        Get filepath based on parameters

    """

    def __init__(
        self,
        dataset: Literal["vanderbilt"],
        subj_ses_select: List[Tuple[str, str]] | None = None,
    ):
        self.subj_ses_select = subj_ses_select
        # load subject list for chosen dataset
        df = pd.read_csv(f"scan/meta/{dataset}_participant.csv", dtype=str)
        # subject and session are necessary fields
        if Field.SUBJECT.value not in df.columns:
            raise ValueError(
                '"subject" column missing from participant list and is a necessary field'
            )
        if Field.SESSION.value not in df.columns:
            # if session is not present, create a list of empty strings
            df[Field.SESSION.value] = ""

        # get location of fields
        self.fields = [Field.SUBJECT.value, Field.SESSION.value]
        self.subject_loc = df.columns.get_loc(Field.SUBJECT.value)
        self.session_loc = df.columns.get_loc(Field.SESSION.value)

        self.dataset = dataset
        # get values for fields from every row in nested list
        all_values = list(zip(*[df[c] for c in self.fields]))
        if self.subj_ses_select is not None:
            self.values = [
                v for v in all_values if (v[0], v[1]) in self.subj_ses_select
            ]
            if len(self.values) == 0:
                raise ValueError(
                    "No matching subjects/sessions found in participant list"
                )
        else:
            self.values = all_values
        self.n_scans = len(self.values)

        # get parameters
        with open("scan/meta/params.json", "rb") as f:
            params = json.load(f)[dataset]
        self.file_format = params["file_format"]

        # check if multiecho dataset
        self.multiecho = params["multiecho"]
        # determine number of echos, if applicable
        if self.multiecho:
            echos = params["func"]["echos"]
            self.n_echos = len(echos)
        else:
            self.n_echos = None

    def __iter__(self):
        """
        iterate through rows of
        subject list in order
        """
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.n_scans:
            current_value = self.values[self.n]
            self.n += 1
            return current_value
        raise StopIteration

    def subject(self) -> Generator[str, None, None]:
        """
        iterate through subjects in order

        Returns
        -------
        subject: str
            subject label
        """
        # does not reorder
        subjects = list(
            dict.fromkeys([s[self.subject_loc] for s in self.values])  # type: ignore
        )
        n_subj = len(subjects)
        n = 0
        while n < n_subj:
            yield subjects[n]
            n += 1

    def subject_by_session(self) -> Generator[Tuple[str, list[str]], None, None]:
        """
        Iterate through subjects and return
        (possible) multiple sessions as a nested
        list. If no session field is present, return
        a list of empty strings.

        Returns
        -------
        subj_sess: Tuple[str, list[str]]
            subject label and list of session labels
        """
        subjects = list(self.subject())
        n_subj = len(subjects)
        n = 0
        while n < n_subj:
            subj = subjects[n]
            if self.session_loc is not None:
                subj_ses = list(
                    dict.fromkeys(
                        [
                            s[self.session_loc]
                            for s in self.values  # type: ignore
                            if s[self.subject_loc] == subj
                        ]  # type: ignore
                    )
                )
            else:
                subj_ses = [""]
            yield (subj, subj_ses)
            n += 1

    def filepath(
        self,
        data: Literal["func", "anat", "eeg", "physio"],
        basedir: str | None = None,
        echo: bool = False,
    ) -> Generator[Union[str, List[str]], None, None]:
        """
        iterate through rows and return file path to
        data object (eeg, anat, func or physio). If specified,
        prepend with a directory path with basedir. If multiecho,
        return for all echo fps as a list by setting echo = True.

        Parameters
        ----------
        data: Literal['func', 'anat', 'eeg', 'physio']
            data modality
        basedir: str
            base directory file path to prepend to file (optional)
        echo: bool
            whethe to return filepaths for individual echos for
            multiecho data (default: False)

        Returns
        -------
        fp: str
            file path
        """
        # initialize counter
        n = 0
        while n < self.n_scans:
            # get values for each field per row
            ss = dict(zip(self.fields, self.values[n]))
            # if specified, loop through echos
            if (data == "func") & echo & self.multiecho:
                fp_echos = []
                for n_e in range(self.n_echos):  # type: ignore
                    # assuming echo label always start at 0
                    # may not always be the case
                    ss["echo"] = n_e + 1
                    # path to file, with echo
                    fp = self.to_file(data, basedir=basedir, **ss)
                    fp_echos.append(fp)
                yield fp_echos
                n += 1
            # if echo not specified
            else:
                fp = self.to_file(data, basedir=basedir, **ss)
                yield fp
                n += 1

    def to_file(
        self,
        data: Literal["func", "anat", "eeg", "physio"],
        subject: str,
        session: str | None = None,
        file_ext: utils.FileExtParams | None = None,
        echo: int | None = None,
        physio: str | None = None,
        physio_type: Literal["raw", "out"] | None = None,
        basedir: str | None = None,
    ) -> str:
        """
        take fields (e.g. subject, session, echo) and return file path.
        If echo is provided, return file path to specific echo. If
        specified, prepend with a directory path with basedir.

        Parameters
        ----------
        data: Literal['func', 'anat', 'eeg', 'physio']
            Choice of data modality
        subject: str
            subject label
        file_ext: FileExtParams
            the file extension for the file path. Must be supplied for
            eeg and physio modalities.
        hemi: Literal[rh, lh]
            left or right hemisphere. Ignored if data modality not 'func'
            (optional)
        session: str
            session label (optional)
        echo: int
            echo label (optional)
        physio: str
            physio label (optional)
        physio_type: Literal['raw', 'out']
            type of physio file to return (optional)
        basedir: str
            prepend filepath to directory (optional)

        Returns
        -------
        fp: str
            file path
        """
        # check if file format is available for data modality
        if data not in self.file_format:
            raise ValueError(f"{data} file format not available for {self.dataset}")
        # check if basedir is provided
        if basedir is not None:
            basepath = basedir + "/"
        else:
            basepath = ""

        # handle file extensions
        file_ext = self._fileext(data, file_ext)  # type: ignore

        if data == "func":
            if self.multiecho:
                if echo is not None:
                    fp = self.file_format[data]["echo"].format(
                        subject=subject, session=session, echo=echo, ext=file_ext
                    )
                else:
                    fp = self.file_format[data]["combined"].format(
                        subject=subject, session=session, ext=file_ext
                    )
            else:
                fp = self.file_format[data].format(
                    subject=subject, session=session, ext=file_ext
                )
        elif data == "physio":
            if physio_type == "raw":
                fp = self.file_format[data]["raw"][physio].format(
                    subject=subject, session=session, physio=physio, ext=file_ext
                )
            elif physio_type == "out":
                fp = self.file_format[data]["out"].format(
                    subject=subject, session=session, physio=physio, ext=file_ext
                )
        else:
            fp = self.file_format[data].format(
                subject=subject, session=session, ext=file_ext
            )
        return f"{basepath}{fp}"

    def _fileext(self, data: str, file_ext: str) -> str:
        """
        handle file extensions for various data modalities
        """
        allowed_func_exts = [
            "nii",
            "nii.gz",
            "lh.func.gii",
            "rh.func.gii",
            "func.gii",
            "dtseries.nii",
        ]

        if data == "func":
            # default extension for functional scans is nii.gz
            if file_ext is None:
                file_ext_out = "nii.gz"
            # check if file extension is allowed
            elif file_ext in allowed_func_exts:
                file_ext_out = file_ext
            # check if func.gii is specified
            elif file_ext == "func.gii":
                raise ValueError("must specify hemisphere (lh,rh) for func.gii")
            else:
                raise ValueError(
                    f"file ext {file_ext} not supported for functional scans"
                )
        elif data in ["eeg", "physio"]:
            if file_ext is None:
                raise ValueError(
                    "file extension must be supplied for eeg/physio file path"
                )
            file_ext_out = file_ext
        else:
            file_ext_out = file_ext

        return file_ext_out
