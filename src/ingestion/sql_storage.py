"""
SQLite Storage pour données Excel
Stocke les données brutes pour requêtes numériques (agrégations, calculs)
Mode OFFLINE compatible
"""

import sqlite3
from pathlib import Path
from typing import Dict, List
import pandas as pd
from loguru import logger


class SQLiteStorage:
    """
    Gestionnaire SQLite pour données Excel
    Permet des requêtes SQL directes sur les tableaux
    
    Usage:
        storage = SQLiteStorage("/app/backups/excel_data.db")
        storage.store_dataframe(df, "factures_janvier", metadata)
        results = storage.execute_query("SELECT SUM(montant) FROM factures_janvier")
    """
    
    # backups/ et non data/ : le dossier de documents est monte en lecture
    # seule par le watcher, SQLite ne peut pas y creer sa base.
    def __init__(self, db_path: str = "/app/backups/excel_data.db"):
        """
        Initialise le stockage SQLite
        
        Args:
            db_path: Chemin vers le fichier de base de données
        """
        self.db_path = db_path
        self._ensure_db_exists()
        logger.info(f"SQLiteStorage initialized: {db_path}")
    
    def _ensure_db_exists(self):
        """Crée le fichier DB et le dossier parent si nécessaire"""
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Créer la table de métadonnées
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _excel_metadata (
                    table_name TEXT PRIMARY KEY,
                    source_file TEXT,
                    file_path TEXT,
                    sheet_name TEXT,
                    columns TEXT,
                    column_types TEXT,
                    row_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def store_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str, 
        metadata: Dict
    ) -> bool:
        """
        Stocke un DataFrame en table SQLite
        
        Args:
            df: DataFrame pandas à stocker
            table_name: Nom de la table SQL
            metadata: Métadonnées de la feuille Excel
            
        Returns:
            True si succès
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Stocker les données
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
                # Préparer column_types comme JSON string
                import json
                column_types_str = json.dumps(metadata.get("column_types", {}))
                
                # Stocker les métadonnées
                conn.execute("""
                    INSERT OR REPLACE INTO _excel_metadata 
                    (table_name, source_file, file_path, sheet_name, columns, column_types, row_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    table_name,
                    metadata.get("file_name", ""),
                    metadata.get("file_path", ""),
                    metadata.get("sheet_name", ""),
                    ",".join(str(c) for c in metadata.get("columns", [])),
                    column_types_str,
                    metadata.get("row_count", len(df))
                ))
                conn.commit()
                
            logger.info(f"Stored {len(df)} rows in SQL table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in SQL: {e}")
            return False
    
    def execute_query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """
        Exécute une requête SQL et retourne un DataFrame
        
        Args:
            sql: Requête SQL SELECT
            params: Paramètres optionnels pour requête préparée
            
        Returns:
            DataFrame avec les résultats
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if params:
                    return pd.read_sql_query(sql, conn, params=params)
                return pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            raise
    
    def list_tables(self) -> List[Dict]:
        """
        Liste toutes les tables Excel stockées
        
        Returns:
            Liste de dictionnaires avec infos sur chaque table
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT table_name, source_file, file_path, sheet_name, 
                           columns, row_count, created_at, updated_at
                    FROM _excel_metadata
                    ORDER BY updated_at DESC
                """)
                return [
                    {
                        "table": row[0],
                        "source": row[1],
                        "file_path": row[2],
                        "sheet": row[3],
                        "columns": row[4].split(",") if row[4] else [],
                        "rows": row[5],
                        "created": row[6],
                        "updated": row[7]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        """
        Retourne le schéma d'une table
        
        Args:
            table_name: Nom de la table
            
        Returns:
            Liste de colonnes avec leur type
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                return [
                    {
                        "name": row[1], 
                        "type": row[2], 
                        "nullable": not row[3],
                        "primary_key": bool(row[5])
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to get schema for {table_name}: {e}")
            return []
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Retourne un échantillon de données d'une table
        
        Args:
            table_name: Nom de la table
            limit: Nombre de lignes max
            
        Returns:
            DataFrame avec l'échantillon
        """
        return self.execute_query(f"SELECT * FROM {table_name} LIMIT ?", (limit,))
    
    def get_table_stats(self, table_name: str) -> Dict:
        """
        Calcule des statistiques sur une table
        
        Args:
            table_name: Nom de la table
            
        Returns:
            Dict avec statistiques par colonne numérique
        """
        try:
            schema = self.get_table_schema(table_name)
            numeric_cols = [
                col["name"] for col in schema 
                if col["type"].upper() in ("INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE")
            ]
            
            stats = {}
            with sqlite3.connect(self.db_path) as conn:
                for col in numeric_cols:
                    cursor = conn.execute(f"""
                        SELECT 
                            MIN({col}) as min_val,
                            MAX({col}) as max_val,
                            AVG({col}) as avg_val,
                            SUM({col}) as sum_val,
                            COUNT({col}) as count_val
                        FROM {table_name}
                    """)
                    row = cursor.fetchone()
                    stats[col] = {
                        "min": row[0],
                        "max": row[1],
                        "avg": row[2],
                        "sum": row[3],
                        "count": row[4]
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for {table_name}: {e}")
            return {}
    
    def delete_table(self, table_name: str) -> bool:
        """
        Supprime une table et ses métadonnées
        
        Args:
            table_name: Nom de la table à supprimer
            
        Returns:
            True si succès
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.execute(
                    "DELETE FROM _excel_metadata WHERE table_name = ?",
                    (table_name,)
                )
                conn.commit()
            logger.info(f"Deleted SQL table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete table: {e}")
            return False
    
    def delete_by_source(self, source_file: str) -> int:
        """
        Supprime toutes les tables provenant d'un fichier source
        
        Args:
            source_file: Nom du fichier source
            
        Returns:
            Nombre de tables supprimées
        """
        try:
            tables = self.list_tables()
            deleted = 0
            for table in tables:
                if table["source"] == source_file:
                    if self.delete_table(table["table"]):
                        deleted += 1
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete by source: {e}")
            return 0
    
    def search_tables(self, query: str) -> List[Dict]:
        """
        Recherche des tables par nom ou source
        
        Args:
            query: Terme de recherche
            
        Returns:
            Tables correspondantes
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT table_name, source_file, sheet_name, row_count
                    FROM _excel_metadata
                    WHERE table_name LIKE ? OR source_file LIKE ? OR sheet_name LIKE ?
                """, (f"%{query}%", f"%{query}%", f"%{query}%"))
                return [
                    {
                        "table": row[0],
                        "source": row[1],
                        "sheet": row[2],
                        "rows": row[3]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to search tables: {e}")
            return []


