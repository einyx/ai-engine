use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::frame::{read_frame, write_frame};
use ai_engine_cluster::transport::quic::{client_endpoint, server_endpoint};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loopback_echo_via_quic_streams() {
    let server_id = generate_node_identity("server").unwrap();
    let server_ep = server_endpoint(&server_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let server_addr = server_ep.local_addr().unwrap();

    let server_task = tokio::spawn(async move {
        let conn = server_ep
            .accept()
            .await
            .expect("accept")
            .await
            .expect("conn");
        let (mut send, mut recv) = conn.accept_bi().await.expect("bi-stream");
        let msg = read_frame(&mut recv).await.expect("read");
        write_frame(&mut send, &msg).await.expect("write");
        send.finish().expect("finish");
        conn.closed().await;
    });

    let client_id = generate_node_identity("client").unwrap();
    let client_ep = client_endpoint(&client_id, std::slice::from_ref(&server_id.fingerprint)).unwrap();
    let conn = client_ep
        .connect(server_addr, "server")
        .unwrap()
        .await
        .expect("connect");
    let (mut send, mut recv) = conn.open_bi().await.expect("open_bi");
    write_frame(&mut send, b"hello world").await.unwrap();
    send.finish().unwrap();
    let echoed = read_frame(&mut recv).await.unwrap();
    assert_eq!(echoed, b"hello world");

    drop(conn);
    let _ = server_task.await;
}
