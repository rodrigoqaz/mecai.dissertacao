import os
import boto3
import awswrangler as wr
import pandas as pd

session = boto3.Session(profile_name = 'locks-prd', region_name='us-east-1')

def consulta_dados_tracecotton(session):

    query = f"""
    WITH classificacoes AS (
        SELECT
            f.codigosai 
            ,ac.descricao AS classificacao
            ,'lote' AS origem
            ,CAST(ml.numero AS varchar) AS numero
            ,am.nome AS maquina_algodoeira
            ,faz.nome AS fazenda_origem
            ,f.databeneficiamento
            ,r.datacolheita
            ,r.periodocolheita
            ,r.bordadura
            ,r.umidade
            ,r.sinistros
            ,r.contaminantes
            ,r.localizacaolatitude
            ,r.localizacaolongitude
            ,r.localizacaoaltitude
            ,t.numero AS talhao
            ,t.idexterno AS idexterno
            ,t.areaha
            ,t.dataplantio
            ,v.nome AS nome_variedade
            ,v.descricao AS descricao_variedade
            ,g.nome AS nome_grupovariedade
            ,g.descricao AS descricao_grupovariedade
        FROM tracecotton.fardinho f
        INNER JOIN tracecotton.algodoeiramaquina am ON (am.algodoeiramaquinaid = f.algodoeiramaquinaid)
        INNER JOIN tracecotton.fazenda faz ON (faz.fazendaid = f.fazendaorigemid)
        INNER JOIN tracecotton.rolo r ON (r.roloid = f.roloid)
        INNER JOIN tracecotton.ordemcolheita oc ON (oc.ordemcolheitaid = r.ordemcolheitaid)
        INNER JOIN tracecotton.talhao t ON (t.talhaoid = oc.talhaoid)
        INNER JOIN tracecotton.variedade v ON (v.variedadeid = t.variedadeid)
        INNER JOIN tracecotton.grupovariedade g ON (g.grupovariedadeid = v.grupovariedadeid)
        INNER JOIN tracecotton.malaamostralote ml ON (ml.malaamostraloteid = f.malaamostraloteid)
        INNER JOIN tracecotton.algodaoclassificacao ac ON (ac.algodaoclassificacaoid = ml.algodaoclassificacaoid)
        
        UNION all

        SELECT
            f.codigosai 
            ,ac.descricao AS classificacao
            ,'mala' AS origem
            ,mai.numero AS numero
            ,am.nome AS maquina_algodoeira
            ,faz.nome AS fazenda_origem
            ,f.databeneficiamento
            ,r.datacolheita
            ,r.periodocolheita
            ,r.bordadura
            ,r.umidade
            ,r.sinistros
            ,r.contaminantes
            ,r.localizacaolatitude
            ,r.localizacaolongitude
            ,r.localizacaoaltitude
            ,t.numero AS talhao
            ,t.idexterno AS idexterno
            ,t.areaha
            ,t.dataplantio
            ,v.nome AS nome_variedade
            ,v.descricao AS descricao_variedade
            ,g.nome AS nome_grupovariedade
            ,g.descricao AS descricao_grupovariedade
        FROM tracecotton.fardinho f
        INNER JOIN tracecotton.algodoeiramaquina am ON (am.algodoeiramaquinaid = f.algodoeiramaquinaid)
        INNER JOIN tracecotton.fazenda faz ON (faz.fazendaid = f.fazendaorigemid)
        INNER JOIN tracecotton.rolo r ON (r.roloid = f.roloid)
        INNER JOIN tracecotton.ordemcolheita oc ON (oc.ordemcolheitaid = r.ordemcolheitaid)
        INNER JOIN tracecotton.talhao t ON (t.talhaoid = oc.talhaoid)
        INNER JOIN tracecotton.variedade v ON (v.variedadeid = t.variedadeid)
        INNER JOIN tracecotton.grupovariedade g ON (g.grupovariedadeid = v.grupovariedadeid)
        INNER JOIN tracecotton.fardinhoamostraintermediaria fai ON (fai.codigosai = f.codigosai AND fai.organizacaoid = f.organizacaoid)
        INNER JOIN tracecotton.malaamostraintermediaria mai ON (fai.malaamostraid = mai.malaamostraintermediariaid)
        INNER JOIN tracecotton.algodaoclassificacao ac ON (ac.algodaoclassificacaoid = mai.algodaoclassificacaoid)  
    ), temp AS (
    SELECT 
        *
        ,ROW_NUMBER() OVER(PARTITION BY codigosai, classificacao ORDER BY codigosai, origem) AS rn
    FROM classificacoes
    )
    SELECT * FROM temp WHERE rn=1
    """
    df = wr.athena.read_sql_query(
            sql=query,
            database='tracecotton',
            ctas_approach=False, 
            boto3_session=session,
            s3_output='s3://locks-query-result/rodrigo.oliveira'
        )

    return df

amostras = []
for root, dirs, files in os.walk('image_capture/amostras/'):
    for file in files:
        arquivo_sem_extensao = os.path.splitext(file)[0]
        codigo_fardinho = arquivo_sem_extensao[3:]
        amostras.append(codigo_fardinho)

dados_fardinhos = consulta_dados_tracecotton(session)
dados_fardinhos.to_parquet('dados_fardinhos.parquet')
amostras_classificadas = dados_fardinhos[dados_fardinhos['codigosai'].isin(amostras)]
print(amostras_classificadas.head())
amostras_classificadas.to_parquet('amostras_classificadas.parquet')
contagem_amostras = pd.Series(amostras).value_counts().to_dict()
amostras_classificadas['qtd_imagens'] = amostras_classificadas['codigosai'].map(contagem_amostras).fillna(0).astype(int)


resultado = amostras_classificadas.groupby('classificacao').agg(
    quantidade_amostras=('codigosai', 'count'), 
    quantidade_imagens=('qtd_imagens', 'sum') 
).reset_index()

print(resultado)
print(f'Total de amostras: {resultado['quantidade_amostras'].sum()}')
print(f'Total de imagens: {resultado['quantidade_imagens'].sum()}')






