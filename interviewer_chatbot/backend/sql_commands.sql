CREATE TABLE public.checkpoint_blobs (
    thread_id     text NOT NULL,
    checkpoint_ns text NOT NULL DEFAULT ''::text,
    channel       text NOT NULL,
    version       text NOT NULL,
    type          text NOT NULL,
    blob          bytea,

    CONSTRAINT checkpoint_blobs_pkey
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

-- Index
CREATE INDEX checkpoint_blobs_thread_id_idx
    ON public.checkpoint_blobs (thread_id);


    CREATE TABLE public.checkpoint_migrations (
    v integer NOT NULL,

    CONSTRAINT checkpoint_migrations_pkey
        PRIMARY KEY (v)
);

CREATE TABLE public.checkpoint_writes (
    thread_id     text    NOT NULL,
    checkpoint_ns text    NOT NULL DEFAULT ''::text,
    checkpoint_id text    NOT NULL,
    task_id       text    NOT NULL,
    idx           integer NOT NULL,
    channel       text    NOT NULL,
    type          text,
    blob          bytea   NOT NULL,
    task_path     text    NOT NULL DEFAULT ''::text,

    CONSTRAINT checkpoint_writes_pkey
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- Index
CREATE INDEX checkpoint_writes_thread_id_idx
    ON public.checkpoint_writes (thread_id);


CREATE TABLE public.checkpoints (
    thread_id            text  NOT NULL,
    checkpoint_ns        text  NOT NULL DEFAULT ''::text,
    checkpoint_id        text  NOT NULL,
    parent_checkpoint_id text,
    type                 text,
    checkpoint           jsonb NOT NULL,
    metadata             jsonb NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT checkpoints_pkey
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Index
CREATE INDEX checkpoints_thread_id_idx
    ON public.checkpoints (thread_id);