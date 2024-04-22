# """ Handling of ZINC datasets. """
#
# from jointformer.utils.datasets.smiles_tokenizers.base import SMILESDataset
#
#
# class ZINC250KDataset(SMILESDataset):
#
#     def _load_data(self, file_path: str):
#         load_zinc15(featurizer: Featurizer | str = 'OneHot', splitter: Splitter | str | None = 'random', transformers:
#         List[TransformerGenerator | str] = [
#             'normalization'], reload: bool = True, data_dir: str | None = None, save_dir: str | None = None, dataset_size: str = '250K', dataset_dimension: str = '2D', tasks:
#         List[str] = ['mwt', 'logp', 'reactive'], ** kwargs) → Tuple[List[str], Tuple[Dataset, ...], List[Transformer]]
#
#
#
# class ZINC15Dataset:
#     pass
#
# class ZincDataModule(SimpleMolListDataModule):
#     """
#     DataModule for Zinc dataset.
#
#     The molecules are read as SMILES from a number of
#     csv files.
#     """
#
#     def get_data(self) -> Dict[str, Any]:
#         return self._all_data
#
#     def _load_all_data(self) -> None:
#         path = Path(self.dataset_path)
#         if path.is_dir():
#             dfs = [pd.read_csv(filename) for filename in path.iterdir()]
#             df = pd.concat(dfs, ignore_index=True, copy=False)
#         else:
#             df = pd.read_csv(path)
#         self._all_data = {"smiles_tokenizers": df["smiles_tokenizers"].tolist()}
#         self._set_split_indices_from_dataframe(df)
