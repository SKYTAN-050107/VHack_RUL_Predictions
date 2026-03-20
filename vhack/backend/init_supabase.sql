-- Enable the pgvector extension to work with embeddings
create extension if not exists vector;

-- Machines table
create table if not exists machines (
  id bigint primary key generated always as identity,
  name text not null,
  type text not null,
  current_rul integer default 0,
  status text default 'Green' check (status in ('Red', 'Yellow', 'Green')),
  last_updated timestamp with time zone default now()
);

-- Sensor readings history (append-only time series)
create table if not exists sensor_readings (
  id bigint primary key generated always as identity,
  machine_id bigint not null references machines(id) on delete cascade,
  source text not null default 'upload' check (source in ('upload', 'simulator')),
  operating_mode text default 'normal' check (operating_mode in ('idle', 'warmup', 'normal', 'stressed', 'maintenance')),
  vibration double precision,
  temperature double precision,
  load double precision,
  anomaly_score double precision default 0,
  recorded_at timestamp with time zone default now()
);

create index if not exists idx_sensor_readings_machine_recorded_at
on sensor_readings(machine_id, recorded_at desc);

-- Prediction history for trend charts and auditability
create table if not exists prediction_history (
  id bigint primary key generated always as identity,
  machine_id bigint not null references machines(id) on delete cascade,
  source text not null default 'upload' check (source in ('upload', 'simulator')),
  dataset_id text not null default 'FD001',
  rul_prediction double precision not null,
  health_state text not null,
  status text not null check (status in ('Red', 'Yellow', 'Green')),
  change_point_detected boolean default false,
  change_point_step integer,
  explanation text,
  predicted_at timestamp with time zone default now()
);

create index if not exists idx_prediction_history_machine_predicted_at
on prediction_history(machine_id, predicted_at desc);

-- SHAP explanations captured on Analyze click
create table if not exists shap_explanations (
  id bigint primary key generated always as identity,
  machine_id bigint not null references machines(id) on delete cascade,
  prediction_id bigint references prediction_history(id) on delete set null,
  source text not null default 'analysis' check (source in ('analysis', 'upload', 'simulator')),
  model_type text not null default 'joblib',
  dataset_id text not null default 'FD001',
  base_value double precision,
  model_output double precision,
  top_features jsonb not null,
  full_values jsonb,
  created_at timestamp with time zone default now()
);

create index if not exists idx_shap_explanations_machine_created_at
on shap_explanations(machine_id, created_at desc);

create index if not exists idx_shap_explanations_prediction_id
on shap_explanations(prediction_id);

-- Machine simulation profile for realistic synthetic telemetry
create table if not exists simulation_profiles (
  id bigint primary key generated always as identity,
  machine_id bigint unique not null references machines(id) on delete cascade,
  base_vibration double precision not null,
  base_temperature double precision not null,
  base_load double precision not null,
  wear_rate double precision not null default 0.004,
  anomaly_probability double precision not null default 0.04,
  dataset_id text not null default 'FD001',
  updated_at timestamp with time zone default now()
);

-- Maintenance logs
create table if not exists maintenance_logs (
  id bigint primary key generated always as identity,
  machine_id bigint references machines(id),
  technician_name text not null,
  status text default 'Active' check (status in ('Active', 'Completed')),
  root_cause text,
  action_taken text,
  steps text,
  components text,
  estimated_time text,
  completion_date timestamp with time zone,
  created_at timestamp with time zone default now()
);

-- Resources (Technical/Financial)
create table if not exists resources (
  id bigint primary key generated always as identity,
  filename text not null,
  resource_type text not null check (resource_type in ('technical', 'financial')),
  uploaded_at timestamp with time zone default now()
);

-- Embeddings for RAG
create table if not exists resource_embeddings (
  id bigint primary key generated always as identity,
  resource_id bigint references resources(id),
  content text,
  metadata jsonb,
  embedding vector(3072) -- Using Gemini embeddings (3072)
);

-- Vector search function for RAG
create or replace function match_resource_embeddings (
  query_embedding vector(3072),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  resource_id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    re.id,
    re.resource_id,
    re.content,
    re.metadata,
    1 - (re.embedding <=> query_embedding) as similarity
  from resource_embeddings re
  where 1 - (re.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
end;
$$;

-- Initial data for testing
insert into machines (name, type, current_rul, status) values 
('Conveyor Belt A', 'Conveyor', 450, 'Green'),
('Motor Pump 03', 'Pump', 120, 'Yellow'),
('Hydraulic Press', 'Press', 15, 'Red');

insert into simulation_profiles (machine_id, base_vibration, base_temperature, base_load, wear_rate, anomaly_probability, dataset_id)
select m.id,
  case when lower(m.type) = 'conveyor' then 0.50 when lower(m.type) = 'pump' then 0.70 else 0.90 end,
  case when lower(m.type) = 'conveyor' then 52.0 when lower(m.type) = 'pump' then 58.0 else 65.0 end,
  case when lower(m.type) = 'conveyor' then 80.0 when lower(m.type) = 'pump' then 95.0 else 120.0 end,
  0.004,
  0.04,
  'FD001'
from machines m
where not exists (
  select 1 from simulation_profiles sp where sp.machine_id = m.id
);
