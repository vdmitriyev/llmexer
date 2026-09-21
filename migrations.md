### 🧬 Migration from a database older than `0.4.6`

`0.4.6` adds two columns to every experiment table: `data_id` (the `ID` of the `data.csv` row) and `prompt_id` (the name of the prompt file, without `.txt`).

Both were already part of `code`, which is built as `<data_id>_<prompt_id>_<model_name>_<profile_name>`, but they were never stored on their own.

A database generated before `0.4.6` does not have the two columns, and `llmexer` will refuses to open them. You have two ways to solve it:

- run `experiment generate` again, which builds a new database and loses the answers already collected;
- migrate the database in place with the SQL below, which keeps results.

### Step 0: make a backup copy of the database

Before the migration, make a copy of the database

> **Hint:** copy the `.db` file before you start. These statements rewrite it in place and there is no undo.

### Step 1: add the two columns

Each provider has an `experiment_<provider>` table, and a database you have used
`experiment try` on also has a `try_experiment_<provider>` table. Both need the columns. The
`params_<provider>`, `try_param_<provider>` and `datafix_logs` tables do not: they hold
parameters and logs, not identity.

Rather than guess which tables your database has, let it tell you. There is one query per
table family. Each prints the `ALTER TABLE` statements you need, and skips any table that
already has the columns.

For the **provider** tables:

```sql
SELECT 'ALTER TABLE "' || m.name || '" ADD COLUMN data_id TEXT;'  || char(10) ||
       'ALTER TABLE "' || m.name || '" ADD COLUMN prompt_id TEXT;' as "SQL"
FROM sqlite_master AS m
WHERE m.type = 'table'
  AND m.name LIKE 'experiment\_%' ESCAPE '\'
  AND NOT EXISTS (SELECT 1 FROM pragma_table_info(m.name) WHERE name = 'data_id')
ORDER BY m.name;
```

Copy its output back into the client and run it. Example:

```sql
ALTER TABLE "experiment_ollama" ADD COLUMN data_id TEXT;
ALTER TABLE "experiment_ollama" ADD COLUMN prompt_id TEXT;
```

For the **try** tables:

```sql
SELECT 'ALTER TABLE "' || m.name || '" ADD COLUMN data_id TEXT;'  || char(10) ||
       'ALTER TABLE "' || m.name || '" ADD COLUMN prompt_id TEXT;' as "SQL"
FROM sqlite_master AS m
WHERE m.type = 'table'
  AND m.name LIKE 'try\_experiment\_%' ESCAPE '\'
  AND NOT EXISTS (SELECT 1 FROM pragma_table_info(m.name) WHERE name = 'data_id')
ORDER BY m.name;
```

Copy its output back into the client and run it. Example:

```sql
ALTER TABLE "try_experiment_ollama" ADD COLUMN data_id TEXT;
ALTER TABLE "try_experiment_ollama" ADD COLUMN prompt_id TEXT;
```

A database you have never run `experiment try` against has no try tables, so the second query
returns nothing. That is fine — there is nothing to migrate.

### Step 2: fill the two columns from `code`

`code` reads `<data_id>_<prompt_id>_<model_name>_<profile_name>`, so `data_id` is the first
token before `_` and `prompt_id` is the second.

As in step 1, this query prints one `UPDATE` per table:

```sql
SELECT 'UPDATE "' || m.name || '"' || char(10) ||
       '   SET data_id   = substr(code, 1, instr(code, ''_'') - 1),' || char(10) ||
       '       prompt_id = substr(substr(code, instr(code, ''_'') + 1), 1,' || char(10) ||
       '                          instr(substr(code, instr(code, ''_'') + 1), ''_'') - 1)' || char(10) ||
       ' WHERE code IS NOT NULL AND instr(code, ''_'') > 1;' as "SQL"
FROM sqlite_master AS m
WHERE m.type = 'table'
  AND (m.name LIKE 'experiment\_%' ESCAPE '\' OR m.name LIKE 'try\_experiment\_%' ESCAPE '\')
ORDER BY m.name;
```

Each statement the SQL query prints should looks like this (run one per table):

```sql
UPDATE "experiment_ollama"
   SET data_id   = substr(code, 1, instr(code, '_') - 1),
       prompt_id = substr(substr(code, instr(code, '_') + 1), 1,
                          instr(substr(code, instr(code, '_') + 1), '_') - 1)
 WHERE code IS NOT NULL AND instr(code, '_') > 1;
```

The `WHERE` skips any row whose `code` has no `_` in it.
Those rows keep `NULL` in both columns,
which is the honest answer: there is nothing in such a `code` to read.


### Step 3: check the result

```sql
SELECT ID, code, data_id, prompt_id FROM "experiment_ollama" LIMIT 5;
SELECT count(*) FROM "experiment_ollama" WHERE data_id IS NULL;
```

The first query should show the two columns matching the start of each `code`. The second counts
the rows the migration could not read; on a database written by `experiment generate` it is `0`.