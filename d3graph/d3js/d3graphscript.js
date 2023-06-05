function d3graphscript(config = {
  // Default values
  width: 800,
  height: 600,
  charge: -250,
  distance: 0,
  directed: false,
  collision: 0.5
}) {
  // Constants for the SVG
  var width = config.width;
  var height = config.height;

  // Set up the colour scale
  var color = d3.scaleOrdinal(d3.schemeCategory20);

  var simulation = d3.forceSimulation()
    .force("charge", d3.forceManyBody().strength(config.charge))
    .force("link", d3.forceLink().distance((d) => d.edge_distance || config.distance))
    .force("center", d3.forceCenter(width / 2, height / 2));

  // DRAGGING START
  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
  }

  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
  }

  var drag = d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);

  // DRAGGING STOP

  // Append an SVG to the body of the HTML page. Assign this SVG as an object to svg
  var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height)
    .call(d3.zoom().on("zoom", function () {
      svg.attr("transform", d3.event.transform);
    }))
    .on("dblclick.zoom", null)
    .append("g");

  var graphRec = JSON.parse(JSON.stringify(graph));

  // force.nodes(graph.nodes)
  //   .on("tick", tick);
simulation.nodes(graph.nodes)
    .on("tick", function() {
      link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

      node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
    });
    

var link = svg.selectAll(".link")
    .data(graph.links)
    .enter().append("line")
    .attr("class", "link")
    .attr('marker-start', function(d){ return 'url(#marker_' + d.marker_start + ')' })
    .attr("marker-end", function(d) {
      if (config.directed) {return 'url(#marker_' + d.marker_end + ')' }})
    .style("stroke-width", function(d) {return d.edge_width;})          // LINK-WIDTH
    .style("stroke", function(d) {return d.color;});      

  var linkText = svg.selectAll(".link-text")
    .data(graph.links)
    .enter().append("text")
    .attr("class", "link-text")
    .attr("font-size", function(d) { return d.label_fontsize + "px"; })
    .style("fill", function(d) { return d.label_color; })
    .style("font-family", "Arial")
    .text(function(d) { return d.label; });

var node = svg.selectAll(".node")
    .data(graph.nodes)
    .enter().append("g")
    .attr("class", "node")
    .call(drag)
    .on('dblclick', connectedNodes); // HIGHLIGHT ON/OFF

  node.append("circle")
    .attr("r", function(d) { return d.node_size; })
    .style("fill", function(d) { return d.node_color; })
    .style("opacity", 0.95)
    .style("stroke-width", function(d) { return d.node_size_edge; })
    .style("stroke", function(d) { return d.node_color_edge; });

  node.append("text")
    .attr("dx", 10)
    .attr("dy", ".35em")
    .text(function(d) { return d.node_name; })
    .style("font-size", function(d) { return d.node_fontsize + "px"; })
    .style("fill", function(d) { return d.node_fontcolor; })
    .style("font-family", "monospace");

  let showInHover = ["node_tooltip"];
  node.append("title")
    .text((d) => Object.keys(d)
      .filter((key) => showInHover.indexOf(key) !== -1)
      .map((key) => `${d[key]}`)
      .join('\n')
    );

  var data_marker = [
    { id: 0, name: 'circle', path: 'M 0, 0  m -5, 0  a 5,5 0 1,0 10,0  a 5,5 0 1,0 -10,0', viewbox: '-6 -6 12 12' },
    { id: 1, name: 'square', path: 'M 0,0 m -5,-5 L 5,-5 L 5,5 L -5,5 Z', viewbox: '-5 -5 10 10' },
    { id: 2, name: 'arrow', path: 'M 0,0 m -5,-5 L 5,0 L -5,5 Z', viewbox: '-5 -5 10 10' },
    { id: 3, name: 'stub', path: 'M 0,0 m -1,-5 L 1,-5 L 1,5 L -1,5 Z', viewbox: '-1 -5 2 10' }
  ];

  svg.append("defs").selectAll("marker")
    .data(data_marker)
    .enter()
    .append('svg:marker')
    .attr('id', function(d) { return 'marker_' + d.name })
    .attr('markerHeight', 10)
    .attr('markerWidth', 10)
    .attr("markerUnits", "userSpaceOnUse")
    .attr('orient', 'auto')
    .attr('refX', 15)
    .attr('refY', 0)
    .attr('viewBox', function(d) { return d.viewbox })
    .append('svg:path')
    .attr('d', function(d) { return d.path })
    .style("fill", '#808080')
    .style("stroke", '#808080')
    .style("opacity", 0.95)
    .style("stroke-width", 1);

  restart();

  function tick() {
    link.attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });

    node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
    linkText.attr("x", function(d) { return (d.source.x + d.target.x) / 2; })
      .attr("y", function(d) { return (d.source.y + d.target.y) / 2; });
  }

  var toggle = 0;
  var linkedByIndex = {};
  for (i = 0; i < graph.nodes.length; i++) {
    linkedByIndex[i + "," + i] = 1;
  }
  graph.links.forEach(function(d) {
    linkedByIndex[d.source.index + "," + d.target.index] = 1;
  });

  function neighboring(a, b) {
    return linkedByIndex[a.index + "," + b.index];
  }

  function connectedNodes() {
    if (toggle == 0) {
      d = d3.select(this).node().__data__;
      node.style("opacity", function(o) {
        return neighboring(d, o) | neighboring(o, d) ? 1 : 0.1;
      });
      link.style("opacity", function(o) {
        return d.index == o.source.index | d.index == o.target.index ? 1 : 0.1;
      });
      toggle = 1;
    } else {
      node.style("opacity", 0.95);
      link.style("opacity", 1);
      toggle = 0;
    }
  }

  // function restart() {
  //   force.nodes(graph.nodes);
  //   force.links(graph.links);
  //   force.start();

  //   link = link.data(graph.links);
  //   link.exit().remove();
  //   link.enter().insert("line", ".node").attr("class", "link");
  //   link.style("stroke-width", function(d) { return d.edge_width; })
  //     .style("stroke", function(d) { return d.color; });

  //   node = node.data(graph.nodes);
  //   node.enter().append("circle").attr("class", "node").attr("r", 5).call(force.drag);
  // }
  function restart() {
    simulation.nodes(graph.nodes);
    simulation.force("link").links(graph.links);
    // simulation is automatically started when nodes or links are assigned

    link = link.data(graph.links);
    link.exit().remove();
    link.enter().insert("line", ".node").attr("class", "link");
    link.style("stroke-width", function(d) { return d.edge_width; })
      .style("stroke", function(d) { return d.color; });

    node = node.data(graph.nodes);
    node.enter().append("circle").attr("class", "node").attr("r", 5).call(d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended));
  }
}
