"""
Excel Parser pour Luciole RAG
- Préserve métadonnées taxonomiques (chemin, nom, contenu)
- Génère Markdown structuré pour Vector DB
- Stocke données brutes en SQLite (optionnel)
- Compatible avec le pipeline d'ingestion existant
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from loguru import logger

try:
    from .sql_storage import SQLiteStorage
    SQL_STORAGE_AVAILABLE = True
except ImportError:
    SQL_STORAGE_AVAILABLE = False
    SQLiteStorage = None


class ExcelParser:
    """
    Parser Excel intégré au pipeline Luciole
    Compatible avec l'interface DocumentParser existante
    
    Fonctionnalités:
    - Lecture de toutes les feuilles d'un classeur
    - Conversion en Markdown structuré avec contexte
    - Stockage optionnel en SQLite pour requêtes numériques
    - Chunking intelligent par nombre de lignes
    - Détection automatique des types de colonnes
    
    Usage:
        parser = ExcelParser()
        result = parser.parse("/path/to/fichier.xlsx")
        # result = {"content": "Markdown...", "metadata": {...}}
    """
    
    def __init__(
        self,
        max_rows_per_chunk: int = 50,
        overlap_rows: int = 5,
        # backups/ et non data/ : le dossier de documents est monte en
        # lecture seule par le watcher, et n'a pas vocation a heberger
        # une base. Surchargeable par settings.yaml (section excel).
        sqlite_path: str = "/app/backups/excel_data.db",
        enable_sql_storage: bool = True,
        include_stats: bool = True,
        detect_key_columns: bool = True
    ):
        """
        Initialise le parser Excel
        
        Args:
            max_rows_per_chunk: Nombre max de lignes par chunk Markdown
            overlap_rows: Chevauchement entre chunks
            sqlite_path: Chemin vers la base SQLite
            enable_sql_storage: Activer le stockage SQL des données brutes
            include_stats: Inclure les statistiques dans le Markdown
            detect_key_columns: Détecter les colonnes clés pour groupement
        """
        self.max_rows_per_chunk = max_rows_per_chunk
        self.overlap_rows = overlap_rows
        self.include_stats = include_stats
        self.detect_key_columns = detect_key_columns
        
        self.enable_sql_storage = enable_sql_storage and SQL_STORAGE_AVAILABLE
        if self.enable_sql_storage:
            self.sql_storage = SQLiteStorage(sqlite_path)
        else:
            self.sql_storage = None
            if enable_sql_storage and not SQL_STORAGE_AVAILABLE:
                logger.warning("SQLiteStorage non disponible - stockage SQL désactivé")
        
        logger.info(
            f"ExcelParser initialized: chunk={max_rows_per_chunk} rows, "
            f"overlap={overlap_rows}, SQL={self.enable_sql_storage}"
        )
    
    @property
    def supported_extensions(self) -> List[str]:
        """Extensions de fichiers supportées"""
        return [".xlsx", ".xls"]
    
    def parse(self, file_path: str) -> Dict:
        """
        Parse un fichier Excel et retourne le contenu avec métadonnées
        Compatible avec l'interface DocumentParser existante
        
        Args:
            file_path: Chemin vers le fichier Excel
            
        Returns:
            Dict avec 'content' et 'metadata' (format standard Luciole)
        """
        logger.info(f"Parsing Excel: {file_path}")
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Extension non supportée: {path.suffix}")
        
        all_sheets_content = []
        all_metadata = {
            "type": "excel",
            "file_name": path.name,
            "file_path": str(file_path),
            "sheets": [],
            "total_rows": 0,
            "total_sheets": 0
        }
        
        try:
            # Charger toutes les feuilles
            excel_file = pd.ExcelFile(file_path)
            all_metadata["total_sheets"] = len(excel_file.sheet_names)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    if df.empty:
                        logger.debug(f"Feuille vide ignorée: {sheet_name}")
                        continue
                    
                    # Nettoyer le DataFrame
                    df = self._clean_dataframe(df)
                    
                    if df.empty:
                        continue
                    
                    # Extraire métadonnées de la feuille
                    sheet_meta = self._extract_sheet_metadata(df, sheet_name, path.name, file_path)
                    all_metadata["sheets"].append(sheet_meta)
                    all_metadata["total_rows"] += len(df)
                    
                    # Convertir en Markdown avec contexte
                    markdown_content = self._dataframe_to_markdown(
                        df, sheet_name, file_path, sheet_meta
                    )
                    all_sheets_content.append(markdown_content)
                    
                    # Stocker en SQL si activé
                    if self.enable_sql_storage and self.sql_storage:
                        table_name = self._generate_table_name(path.stem, sheet_name)
                        success = self.sql_storage.store_dataframe(df, table_name, {
                            **sheet_meta,
                            "file_name": path.name,
                            "file_path": str(file_path)
                        })
                        if success:
                            sheet_meta["sql_table"] = table_name
                            sheet_meta["has_sql_copy"] = True
                    
                    logger.debug(f"Feuille parsée: {sheet_name} ({len(df)} lignes)")
                    
                except Exception as e:
                    logger.warning(f"Erreur parsing feuille {sheet_name}: {e}")
                    continue
            
            excel_file.close()
            
        except Exception as e:
            logger.error(f"Erreur parsing Excel {file_path}: {e}")
            raise
        
        # Combiner tout le contenu
        if not all_sheets_content:
            logger.warning(f"Aucun contenu extrait de: {file_path}")
            return {
                "content": "",
                "metadata": all_metadata
            }
        
        full_content = "\n\n---\n\n".join(all_sheets_content)
        
        logger.info(
            f"Excel parsé: {path.name} - {all_metadata['total_sheets']} feuilles, "
            f"{all_metadata['total_rows']} lignes"
        )
        
        return {
            "content": full_content,
            "metadata": all_metadata
        }
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie un DataFrame
        - Supprime lignes/colonnes entièrement vides
        - Nettoie les noms de colonnes
        - Convertit les types
        """
        # Supprimer colonnes sans nom ou entièrement NaN
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, df.columns.notna()]
        
        # Supprimer lignes entièrement vides
        df = df.dropna(how='all')
        
        # Nettoyer noms de colonnes
        df.columns = [
            str(col).strip() if pd.notna(col) else f"Colonne_{i}"
            for i, col in enumerate(df.columns)
        ]
        
        # Supprimer doublons dans noms de colonnes
        seen = {}
        new_cols = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        
        return df
    
    def _extract_sheet_metadata(
        self, 
        df: pd.DataFrame, 
        sheet_name: str, 
        file_name: str,
        file_path: str
    ) -> Dict:
        """
        Extrait les métadonnées structurées d'une feuille
        
        Args:
            df: DataFrame de la feuille
            sheet_name: Nom de la feuille
            file_name: Nom du fichier
            file_path: Chemin complet
            
        Returns:
            Dict avec métadonnées
        """
        # Détecter types de colonnes
        column_types = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if "int" in dtype:
                column_types[col] = "integer"
            elif "float" in dtype:
                column_types[col] = "decimal"
            elif "datetime" in dtype or "date" in dtype.lower():
                column_types[col] = "datetime"
            elif "bool" in dtype:
                column_types[col] = "boolean"
            else:
                # Essayer de détecter dates en string
                if self._looks_like_date_column(df[col]):
                    column_types[col] = "datetime"
                elif self._looks_like_numeric_column(df[col]):
                    column_types[col] = "numeric_text"
                else:
                    column_types[col] = "text"
        
        # Calculer statistiques pour colonnes numériques
        numeric_stats = {}
        if self.include_stats:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols:
                try:
                    numeric_stats[col] = {
                        "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                        "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                        "mean": round(float(df[col].mean()), 2) if pd.notna(df[col].mean()) else None,
                        "sum": round(float(df[col].sum()), 2) if pd.notna(df[col].sum()) else None
                    }
                except Exception:
                    pass
        
        # Détecter colonnes clés potentielles
        key_columns = []
        if self.detect_key_columns:
            key_columns = self._detect_key_columns(df)
        
        return {
            "sheet_name": sheet_name,
            "columns": list(df.columns),
            "column_types": column_types,
            "row_count": len(df),
            "numeric_stats": numeric_stats,
            "key_columns": key_columns,
            "has_sql_copy": False,
            "sql_table": None
        }
    
    def _looks_like_date_column(self, series: pd.Series, sample_size: int = 10) -> bool:
        """Vérifie si une colonne ressemble à des dates"""
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
            r'\d{2}/\d{2}/\d{4}',  # 15/01/2024
            r'\d{2}-\d{2}-\d{4}',  # 15-01-2024
        ]
        
        sample = series.dropna().head(sample_size)
        if len(sample) == 0:
            return False
        
        matches = 0
        for val in sample:
            val_str = str(val)
            for pattern in date_patterns:
                if re.match(pattern, val_str):
                    matches += 1
                    break
        
        return matches / len(sample) > 0.5
    
    def _looks_like_numeric_column(self, series: pd.Series, sample_size: int = 10) -> bool:
        """Vérifie si une colonne texte contient des nombres"""
        sample = series.dropna().head(sample_size)
        if len(sample) == 0:
            return False
        
        numeric_count = 0
        for val in sample:
            try:
                val_str = str(val).replace(" ", "").replace(",", ".")
                float(val_str)
                numeric_count += 1
            except ValueError:
                pass
        
        return numeric_count / len(sample) > 0.7
    
    def _detect_key_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Détecte les colonnes clés potentielles (pour groupement intelligent)
        Critères: peu de valeurs uniques, pas trop de NaN
        """
        key_columns = []
        
        for col in df.columns:
            try:
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
                null_ratio = df[col].isna().sum() / len(df) if len(df) > 0 else 1
                
                # Colonne clé: peu de valeurs uniques (< 20%) et peu de NaN (< 10%)
                if unique_ratio < 0.2 and null_ratio < 0.1 and df[col].nunique() > 1:
                    key_columns.append(col)
            except Exception:
                pass
        
        return key_columns[:3]  # Max 3 colonnes clés
    
    def _dataframe_to_markdown(
        self, 
        df: pd.DataFrame, 
        sheet_name: str, 
        file_path: str, 
        metadata: Dict
    ) -> str:
        """
        Convertit un DataFrame en Markdown structuré
        Inclut le contexte fichier pour enrichir l'embedding
        
        Args:
            df: DataFrame à convertir
            sheet_name: Nom de la feuille
            file_path: Chemin du fichier
            metadata: Métadonnées de la feuille
            
        Returns:
            Contenu Markdown avec contexte
        """
        path = Path(file_path)
        
        # Extraire le chemin parent (2-3 derniers dossiers)
        parts = path.parts
        if len(parts) > 3:
            parent_path = "/".join(parts[-3:-1])
        elif len(parts) > 1:
            parent_path = str(path.parent.name)
        else:
            parent_path = ""
        
        # Construire l'en-tête avec contexte (format Luciole)
        columns_str = ", ".join(metadata["columns"][:10])
        if len(metadata["columns"]) > 10:
            columns_str += f" (+{len(metadata['columns']) - 10} autres)"
        
        header = f"""## Tableau: {sheet_name}

**Source**: {path.name}
**Chemin**: {parent_path}
**Feuille**: {sheet_name}
**Colonnes**: {columns_str}
**Lignes**: {metadata['row_count']}

"""
        
        # Générer le tableau Markdown
        if len(df) > self.max_rows_per_chunk:
            # Chunking: prendre les premières lignes + message
            df_display = df.head(self.max_rows_per_chunk)
            table_md = self._generate_markdown_table(df_display)
            table_md += f"\n\n*... ({len(df) - self.max_rows_per_chunk} lignes supplémentaires)*"
        else:
            table_md = self._generate_markdown_table(df)
        
        # Ajouter statistiques si activé
        stats_section = ""
        if self.include_stats and metadata.get("numeric_stats"):
            stats_section = "\n\n**Statistiques colonnes numériques**:\n"
            for col, stats in metadata["numeric_stats"].items():
                if stats.get("sum") is not None:
                    stats_section += (
                        f"- **{col}**: somme={stats['sum']:,.2f}, "
                        f"moy={stats['mean']:,.2f}, "
                        f"min={stats['min']:,.2f}, max={stats['max']:,.2f}\n"
                    )
        
        # Ajouter info SQL si disponible
        sql_section = ""
        if metadata.get("has_sql_copy") and metadata.get("sql_table"):
            sql_section = f"\n\n*Table SQL disponible: `{metadata['sql_table']}`*"
        
        return header + table_md + stats_section + sql_section
    
    def _generate_markdown_table(self, df: pd.DataFrame) -> str:
        """
        Génère un tableau Markdown depuis un DataFrame
        Utilise to_markdown() si disponible, sinon génération manuelle
        """
        try:
            # Essayer d'utiliser to_markdown (nécessite tabulate)
            return df.to_markdown(index=False)
        except Exception:
            # Fallback: génération manuelle
            return self._manual_markdown_table(df)
    
    def _manual_markdown_table(self, df: pd.DataFrame) -> str:
        """Génère un tableau Markdown manuellement"""
        cols = list(df.columns)
        
        # En-tête
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        
        # Séparateur
        separator = "|" + "|".join(["---"] * len(cols)) + "|"
        
        # Lignes
        rows = []
        for _, row in df.iterrows():
            values = []
            for v in row.values:
                if pd.isna(v):
                    values.append("")
                elif isinstance(v, float):
                    values.append(f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4f}")
                else:
                    values.append(str(v).replace("|", "\\|"))
            row_str = "| " + " | ".join(values) + " |"
            rows.append(row_str)
        
        return header + "\n" + separator + "\n" + "\n".join(rows)
    
    def _generate_table_name(self, file_stem: str, sheet_name: str) -> str:
        """
        Génère un nom de table SQL valide
        
        Args:
            file_stem: Nom du fichier sans extension
            sheet_name: Nom de la feuille
            
        Returns:
            Nom de table SQL valide
        """
        name = f"{file_stem}_{sheet_name}"
        
        # Nettoyer caractères spéciaux
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        
        # Limite SQLite/PostgreSQL
        return name.lower()[:63]
    
    def chunk_dataframe(
        self, 
        df: pd.DataFrame, 
        key_column: str = None
    ) -> List[Tuple[pd.DataFrame, str]]:
        """
        Découpe un DataFrame en chunks pour ingestion
        
        Args:
            df: DataFrame à découper
            key_column: Colonne pour groupement (optionnel)
            
        Returns:
            Liste de (chunk_df, range_description)
        """
        if len(df) <= self.max_rows_per_chunk:
            return [(df, f"1-{len(df)}")]
        
        chunks = []
        
        if key_column and key_column in df.columns:
            # Groupement par colonne clé
            for key_value, group_df in df.groupby(key_column):
                if len(group_df) > self.max_rows_per_chunk:
                    # Sous-chunker les gros groupes
                    for i in range(0, len(group_df), self.max_rows_per_chunk - self.overlap_rows):
                        end = min(i + self.max_rows_per_chunk, len(group_df))
                        chunk = group_df.iloc[i:end]
                        chunks.append((chunk, f"{key_column}={key_value}, lignes {i+1}-{end}"))
                else:
                    chunks.append((group_df, f"{key_column}={key_value}"))
        else:
            # Chunking séquentiel avec overlap
            for i in range(0, len(df), self.max_rows_per_chunk - self.overlap_rows):
                end = min(i + self.max_rows_per_chunk, len(df))
                chunk = df.iloc[i:end]
                chunks.append((chunk, f"lignes {i+1}-{end}"))
                
                if end >= len(df):
                    break
        
        return chunks
